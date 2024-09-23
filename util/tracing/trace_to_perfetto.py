#!/usr/bin/env python3

import argparse
import json
import typing
from concurrent.futures import ProcessPoolExecutor

from util.tracing.annotation import (
    BarrierEventGenerator,
    DMAEventGenerator,
    StreamingEventGenerator,
    calculate_sections,
)
from util.tracing.state import get_trace_state


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
