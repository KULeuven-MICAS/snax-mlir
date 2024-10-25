import itertools
import json
import pathlib
from io import StringIO

import pandas as pd
from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import arith, builtin, func, linalg, transform
from xdsl.dialects.builtin import i8, i32
from xdsl.ir import Block, BlockArgument, Region
from xdsl.printer import Printer

from util.snax_benchmark import SNAXBenchmark

if __name__ == "__main__":
    """Runs the gendata.py script with specified arguments."""

    for value in (1,):

        folder = f"toy-{value}"
        bm = SNAXBenchmark(
            kernel=f"toy-{value}",
            binary="toy.x",
            src_dir=str(pathlib.Path.cwd()),
            export_dir=str(pathlib.Path.cwd()),
            output_dir=str(pathlib.Path.cwd()),
        )
        bm.clean()
        bm.build(build_opts=[])
        bm.run()
        bm.trace()
        bm.process_traces(folder)
        bm.copy_binary(folder)
        bm.copy_logs(folder)
        bm.copy_results()

