from dataclasses import dataclass, field

from util.tracing.annotation import EventGenerator
from util.tracing.state import TraceState, CSRInstruction

from util.tracing.event import *


from compiler.accelerators.snax_gemm import SNAXGEMMAccelerator
from compiler.dialects.accfg import AcceleratorOp


@dataclass
class _CurrentEvent:
    start: int


@dataclass
class SettingUp(_CurrentEvent):
    number_setups: int = field(default=1)
    number_of_launches: int = field(default=0)
    """
    Since we only end this region when we see the *second* write to launch, we need to keep track of the number 
    of launches
    """


@dataclass
class Launched(_CurrentEvent):
    number_of_zero_writes: int = field(default=0)
    """
    How many zero writes to the launch address have happened
    """
    number_setups: int = field(default=0)


@dataclass
class Stalled(_CurrentEvent):
    pass


class SNAXAcceleratorEventGenerator(EventGenerator):
    """
    Works for the current version of SNAX.

    We emit a "pre-launch" event, that:
        - starts at the first CSR write to a config register
        - ends after all launch fields have been written to
        - records the number of setup instructions executed
    We emit a "launched but not waiting" event, that:
        - starts immediately after the "pre-launch" event
        - ends on the second "write 0 to launch addr"
    We emit a "waiting" event, that:
        - starts on the first read from the barrier address
        - ends when that read return 1


    """

    state: _CurrentEvent | None
    acc: AcceleratorOp
    fields: set[int]
    launch_fields: set[int]
    barrier_addr: int

    def __init__(self):
        super().__init__()
        self.state = None
        self.acc = SNAXGEMMAccelerator().generate_acc_op()
        self.fields = {val.value.data for val in self.acc.fields.data.values()}
        self.launch_fields = {
            val.value.data for val in self.acc.launch_fields.data.values()
        }
        self.barrier_addr = self.acc.barrier.value.data

    def cycle(self, state: TraceState) -> list[DurationEvent]:
        ins = state.instruction
        if not isinstance(ins, CSRInstruction):
            return []
        events: list[DurationEvent] = []

        # if state is None, check if we are setting up, if so, switch state to "setting up"
        if self.state is None:
            if ins.csr in self.fields or self.launch_fields:
                self.state = SettingUp(
                    start=state.clock_cycle,
                )
        # if we are in setup region:
        elif isinstance(self.state, SettingUp):
            # if it's a normal setup, count a setup ins
            if ins.csr in self.fields:
                self.state.number_setups += 1
            # if it's a launch insn
            elif ins.csr in self.launch_fields:
                # check that we haven't met the launch threshold yet
                if self.state.number_of_launches < len(self.launch_fields):
                    self.state.number_of_launches += 1
                    # this launch counts as a setup
                    # self.state.number_setups += 1
                # otherwise, end this event, switch to launch event
                else:
                    events.append(
                        DurationEvent(
                            "setup",
                            self.state.start,
                            state.clock_cycle - self.state.start,
                            "snax",
                            {"setup_ins_count": self.state.number_setups},
                        )
                    )
                    # change state to launched
                    self.state = Launched(state.clock_cycle)
        # if we are launched:
        elif isinstance(self.state, Launched):
            # if we see a setup ins, count it
            if ins.csr in self.fields:
                self.state.number_setups += 1
            # if we see a write to a launch
            elif ins.csr in self.launch_fields:
                self.state.number_of_zero_writes += 1
            if self.state.number_of_zero_writes > 1:
                events.append(
                    DurationEvent(
                        "launched",
                        self.state.start,
                        state.clock_cycle - self.state.start,
                        "snax",
                        {"setup_ins_count": self.state.number_setups},
                    )
                )
                self.state = Stalled(state.clock_cycle)
        elif isinstance(self.state, Stalled):
            # stalled state is resolved with the next CSR ins
            events.append(
                DurationEvent(
                    "stalled",
                    self.state.start - self.state.start,
                    state.clock_cycle,
                    "snax",
                    {},
                )
            )
            self.state = None

        return events
