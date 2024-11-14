#!/usr/bin/env python3

import argparse
import json
import typing
from concurrent.futures import ProcessPoolExecutor

from compiler.accelerators.registry import AcceleratorRegistry
from compiler.accelerators.snax import SNAXAccelerator
from util.tracing.annotation import (
    BarrierEventGenerator,
    DMAEventGenerator,
    StreamingEventGenerator,
    calculate_sections,
)
from util.tracing.snax_event_generator import SNAXAcceleratorEventGenerator
from util.tracing.state import get_trace_state


def worker(file: str, accelerator: str):
    events = []
    generators = [
        BarrierEventGenerator(),
        StreamingEventGenerator(),
        DMAEventGenerator(),
    ]
    if accelerator is not None:
        accelerator_op = (
            AcceleratorRegistry()
            .registered_accelerators[accelerator]()
            .generate_acc_op()
        )
        generators.append(SNAXAcceleratorEventGenerator(accelerator_op))

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
        required=True,
        help='Input performance metric dumps ("*.trace.json")',
    )
    parser.add_argument(
        "--traces",
        metavar="<trace>",
        nargs="+",
        required=True,
        help='Simulation traces to process ("*.dasm")',
    )
    parser.add_argument(
        "--addr2line",
        default="llvm-addr2line",
        help="llvm-addr2line from quidditch toolchain",
    )

    # Only allow SNAX accelerators for now
    snax_accelerators = []
    for accelerator, acc_class in AcceleratorRegistry().registered_accelerators.items():
        if issubclass(acc_class, SNAXAccelerator):
            snax_accelerators.append(accelerator)

    parser.add_argument(
        "--accelerator",
        choices=snax_accelerators,
        default=None,
        help="SNAX accelerator for SNAX Event Annotator",
    )
    parser.add_argument("--elf", help="ELF from which the traces were generated")
    parser.add_argument(
        "-o",
        "--output",
        metavar="<json>",
        type=argparse.FileType("w"),
        nargs="?",
        default="events.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--annotate-kernels",
        action="store_true",
        help="Annotate sections between mcycle with llvm-addr2line, requires --elf and --addr2line",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        default=False,
        help="Don't use ProcessPoolExecutor for parallelizing tracing, useful for debugging",
    )

    args = parser.parse_args()
    # Custom validation: If `--annotate-trace` is set, then `--elf` and `--addr2line` are required
    if args.annotate_kernels and (args.elf is None or args.addr2line is None):
        parser.error("--elf and --addr2line are required when using --annotate-trace")

    return args


# Function that encapsulates the main logic
def process_traces(
    inputs: list[typing.IO[bytes]],
    traces: list[str],
    elf: str,
    addr2line: str,
    accelerator: str,
    output: typing.IO[str] | None = None,
    annotate_kernels: bool = False,
    sequential: bool = False,
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

    if not sequential:
        # Submit trace processing tasks to the executor
        for hartid, file in enumerate(traces):
            futures.append(executor.submit(worker, file, accelerator))
    else:
        # Process traces sequentially for debuggin purposes
        for hartid, file in enumerate(traces):
            result = worker(file, accelerator)
            futures.append(result)

    # Calculate events using provided inputs and arguments
    if annotate_kernels:
        events = calculate_sections(inputs, elf, addr2line, traces)
    else:
        events = []

    if not sequential:
        # Collect results from the executor and convert events to the desired format
        for hartid, f in enumerate(futures):
            events += map(lambda e: e.to_chrome_tracing(hartid), f.result())
    else:
        # Run sequentially for debugging purposes
        for hartid, result in enumerate(
            futures
        ):  # `result` is already the output of `worker`
            events += map(lambda e: e.to_chrome_tracing(hartid), result)

    # Create and write the TraceViewer JSON object
    json.dump({"traceEvents": events, "displayTimeUnit": "ns"}, output, indent=2)

    # Close the output file if it was opened inside the function
    if output.name == "events.json":
        output.close()


# If the script is run directly, use command-line arguments
if __name__ == "__main__":
    args = parse_arguments()
    process_traces(
        args.inputs,
        args.traces,
        args.elf,
        args.addr2line,
        args.accelerator,
        args.output,
        args.annotate_kernels,
        args.sequential,
    )
