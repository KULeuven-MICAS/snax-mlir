#!/usr/bin/env python3

import argparse
import functools
import json
import re
import subprocess
import typing
from concurrent.futures import ProcessPoolExecutor

from util.tracing.annotation import (
    BarrierEventGenerator,
    DMAEventGenerator,
    KernelEvent,
    StreamingEventGenerator,
)
from util.tracing.state import get_trace_state


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


def worker(file: str):
    events = []
    generators = [
        BarrierEventGenerator(),
        StreamingEventGenerator(),
        DMAEventGenerator(),
    ]
    with open(file) as f:
        for index, l in enumerate(f):
            state = get_trace_state(l)
            if state is None:
                raise RuntimeError("Failed to parse trace: " + l)

            for g in generators:
                events += g.check_accumulator(state)
                events += g.cycle(state)
    return events


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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--inputs",
        metavar="<inputs>",
        type=argparse.FileType("rb"),
        nargs="+",
        help="Input performance metric dumps",
    )
    parser.add_argument(
        "--traces", metavar="<trace>", nargs="*", help="Simulation traces to process"
    )
    parser.add_argument("--elf", help="ELF from which the traces were generated")
    parser.add_argument("--addr2line", help="llvm-addr2line from quidditch toolchain")
    parser.add_argument(
        "-o",
        "--output",
        metavar="<json>",
        type=argparse.FileType("w"),
        nargs="?",
        default="events.json",
        help="Output JSON file",
    )
    return parser.parse_args()


# Function that encapsulates the main logic
def process_traces(
    inputs: list[typing.IO[bytes]],
    traces: list[str],
    elf: str,
    addr2line: str,
    output: typing.IO[str] | None = None,
):
    """
    Main processing function that calculates sections and processes traces.

    Args:
        inputs (List[IO[bytes]]): List of input performance metric files opened in binary read mode.
        traces (List[str]): List of simulation trace files to process.
        elf (Optional[str]): ELF file from which the traces were generated.
        addr2line (Optional[str]): Path to the llvm-addr2line tool.
        output (Optional[IO[str]]): Output file to write the JSON result. Defaults to "events.json" if None.
    """
    # Set default output if not provided
    output = output or open("events.json", "w")

    # Create a ProcessPoolExecutor to handle concurrent trace processing
    executor = ProcessPoolExecutor()
    futures = []

    # Submit trace processing tasks to the executor
    for hartid, file in enumerate(traces):
        futures.append(executor.submit(worker, file))

    # Calculate events using provided inputs and arguments
    events = calculate_sections(inputs, elf, addr2line, traces)

    # Collect results from the executor and convert events to the desired format
    for hartid, f in enumerate(futures):
        events += map(lambda e: e.to_chrome_tracing(hartid), f.result())

    # Create and write the TraceViewer JSON object
    json.dump({"traceEvents": events, "displayTimeUnit": "ns"}, output, indent=2)

    # Close the output file if it was opened inside the function
    if output.name == "events.json":
        output.close()


# If the script is run directly, use command-line arguments
if __name__ == "__main__":
    args = parse_arguments()
    process_traces(args.inputs, args.traces, args.elf, args.addr2line, args.output)
