import functools
import json
import re
import subprocess
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

from util.tracing.event import (
    BarrierEvent,
    DMAEvent,
    Event,
    KernelEvent,
    StreamingEvent,
)
from util.tracing.state import (
    CSRInstruction,
    DMCPYIInstruction,
    DMDSTInstruction,
    DMREPInstruction,
    DMSRCInstruction,
    DMSTATIInstruction,
    DMSTRInstruction,
    TraceState,
    get_trace_state,
)


class EventGenerator(ABC):
    def __init__(self):
        self._acc_callback = []

    def schedule_writeback(
        self, state: TraceState, callback: typing.Callable[[int], list[Event]]
    ) -> list[Event]:
        if state.cpu_state["write_rd"] and state.cpu_state["rd"] != 0:
            # The result is written back in the same cycle.
            return callback(state.cpu_state["writeback"])

        # Result will only be assigned at a later cycle (FIFO style).
        self._acc_callback.append(callback)
        return []

    def check_accumulator(self, state: TraceState):
        if not self._acc_callback:
            return []

        if not state.cpu_state["retire_acc"] or state.cpu_state["acc_pid"] == 0:
            return []

        front = self._acc_callback.pop(0)
        return front(state.cpu_state["acc_pdata_32"])

    @abstractmethod
    def cycle(self, state: TraceState) -> list[dict]:
        ...


class BarrierEventGenerator(EventGenerator):
    _INSTANT_THRESHOLD = 10
    _barrier_start_state: TraceState | None = None

    def cycle(self, state: TraceState) -> list[BarrierEvent]:
        result = []

        if self._barrier_start_state is not None:
            # In sequencer.
            if state.pc is None:
                return result

            # Don't bother for instantly succeeding barriers.
            if (
                state.clock_cycle - self._barrier_start_state.clock_cycle
                > self._INSTANT_THRESHOLD
            ):
                result.append(
                    BarrierEvent(
                        self._barrier_start_state.clock_cycle,
                        state.clock_cycle - self._barrier_start_state.clock_cycle,
                        state.pc,
                    )
                )

            self._barrier_start_state = None

        csr_addr = state.cpu_state.get("csr_addr")
        if csr_addr != 0x7C2:
            return result

        self._barrier_start_state = state
        return result


class StreamingEventGenerator(EventGenerator):
    _stream_start_state: TraceState | None = None

    def cycle(self, state: TraceState) -> list[StreamingEvent]:
        result = []

        if (
            not isinstance(state.instruction, CSRInstruction)
            or state.instruction.csr != 0x7C0
        ):
            return result

        # Stream enables and disables always appear twice. Once when issued from the integer core, second when processed
        # by the FPSS. We denote the start as when issued by the integer core and the end when disabled in the FPSS.
        if state.instruction.is_csrrsi:
            if self._stream_start_state is None:
                self._stream_start_state = state
        elif state.instruction.is_csrrci and state.in_fpss_sequencer:
            if self._stream_start_state is not None:
                result.append(
                    StreamingEvent(
                        self._stream_start_state.clock_cycle,
                        state.clock_cycle - self._stream_start_state.clock_cycle,
                    )
                )

            self._stream_start_state = None

        return result


class DMAEventGenerator(EventGenerator):
    @dataclass
    class _InFlightDMA:
        started: int
        source: int
        destination: int
        inner_loop: int
        is_2d: bool
        source_strides: int = 0
        dest_strides: int = 0
        outer_loop: int = 0
        transfer_id: int = 0

    _current_source = 0
    _current_destination = 0
    _current_repetition = 0
    _current_source_stride = 0
    _current_dest_stride = 0
    _in_flight: list[_InFlightDMA]

    def __init__(self):
        super().__init__()
        self._in_flight = []

    @staticmethod
    def _in_flight_to_event(end_observed: int, transfers: list[_InFlightDMA]):
        result = []
        for t in transfers:
            result.append(
                DMAEvent(
                    t.started,
                    end_observed - t.started,
                    t.source,
                    t.destination,
                    t.inner_loop,
                    t.transfer_id,
                    t.is_2d,
                    t.source_strides,
                    t.dest_strides,
                    t.outer_loop,
                )
            )
        return result

    def cycle(self, state: TraceState) -> list[StreamingEvent]:
        result = []

        instruction = state.instruction
        if isinstance(instruction, DMSRCInstruction):
            self._current_source = instruction.read_source(state)
        elif isinstance(instruction, DMDSTInstruction):
            self._current_destination = instruction.read_destination(state)
        elif isinstance(instruction, DMREPInstruction):
            self._current_repetition = instruction.read_reps(state)
        elif isinstance(instruction, DMSTRInstruction):
            self._current_source_stride = instruction.read_source_strides(state)
            self._current_dest_stride = instruction.read_dest_strides(state)
        elif isinstance(instruction, DMCPYIInstruction):
            if instruction.is_2d:
                o = self._InFlightDMA(
                    state.clock_cycle,
                    self._current_source,
                    self._current_destination,
                    instruction.read_size(state),
                    is_2d=True,
                    source_strides=self._current_source_stride,
                    dest_strides=self._current_dest_stride,
                    outer_loop=self._current_repetition,
                )
            else:
                o = self._InFlightDMA(
                    state.clock_cycle,
                    self._current_source,
                    self._current_destination,
                    instruction.read_size(state),
                    is_2d=False,
                )

            def write_back(txid):
                o.transfer_id = txid
                return []

            result += self.schedule_writeback(state, write_back)
            self._in_flight.append(o)

        elif isinstance(instruction, DMSTATIInstruction):
            if instruction.status == 2:

                def write_back(is_busy):
                    inner_result = []
                    if not is_busy:
                        inner_result += self._in_flight_to_event(
                            state.clock_cycle, self._in_flight
                        )
                        self._in_flight.clear()
                    return inner_result

                result += self.schedule_writeback(state, write_back)

        return result


class KernelNameResolver:
    elf: str
    addr2line: str
    traces: tuple[typing.IO]

    def __init__(self, elf, addr2line, traces: tuple[str]):
        self.elf = elf
        self.addr2line = addr2line
        self.traces = (*map(lambda t: open(t), traces),)

    def __exit__(self, exc_type, exc_val, exc_tb):
        map(lambda t: t.close(), self.traces)

    @functools.cache
    def get_name_from_address(self, address: int):
        p = subprocess.run(
            [self.addr2line, "-e", self.elf, "-f", hex(address)],
            check=True,
            capture_output=True,
            text=True,
        )
        return p.stdout.splitlines()[0]

    def get_name(self, cycle: int, hartid: int):
        pattern = re.compile(r"\s*[0-9]+ " + str(cycle))
        iterator = self.traces[hartid]
        for l in iterator:
            if pattern.match(l):
                break
        else:
            return "<unknown-kernel>"

        for index, l in enumerate(iterator):
            # Give up.
            if index == 100:
                return "<unknown-kernel>"

            res = get_trace_state(l)
            if res is None:
                return "<unknown-kernel>"

            if not res.cpu_state["stall"] and res.cpu_state["pc_d"] != res.pc + 4:
                return self.get_name_from_address(res.cpu_state["pc_d"])


def calculate_sections(
    inputs: typing.Iterable[typing.IO],
    elf: str,
    addr2line: str,
    traces: typing.Iterable[str],
) -> list[dict]:
    resolver = KernelNameResolver(elf, addr2line, (*traces,))

    # TraceViewer events
    events = []
    origin = None
    for hartid, file in enumerate(inputs):
        is_dm_core = hartid == 1

        j = json.load(file)
        for index, section in enumerate(j):
            # In the case of the DMA core we are interested in time spent between kernels to measure overhead of
            # IREE abstractions.
            # if not is_dm_core:
            #    if index % 2 == 0:
            #        continue
            # else:
            #    if index % 2 == 1:
            #        continue

            name = "vm"
            if not is_dm_core:
                name = resolver.get_name(section["start"], hartid)
                origin = "xDSL" if name.endswith("$iree_to_xdsl") else "LLVM"
                name = name.removesuffix("$iree_to_xdsl")

            start = section["start"]
            dur = section["end"] - section["start"]
            events.append(
                KernelEvent(
                    name, start, dur, is_dm_core, origin, section
                ).to_chrome_tracing(hartid)
            )

    return events
