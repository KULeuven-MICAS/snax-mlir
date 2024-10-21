import itertools
import json
import pathlib
from io import StringIO
from pprint import pprint

import pandas as pd
from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import arith, builtin, func, linalg, transform
from xdsl.dialects.builtin import i8, i32
from xdsl.ir import Block, BlockArgument, Region
from xdsl.printer import Printer

from util.snax_benchmark import SNAXBenchmark


def create_matrix_multiply(m, n, k, tiling_factors):
    """
    Generate IR for a matmul in the form of:
    ```
    builtin.module {
        func.func @streamer_matmul(
            %arg_a : memref<16x16xi8>,
            %arg_b : memref<16x16xi32>,
            %arg_c : memref<16x16xi32>) -> tensor<16x16xi32>{
        %0 = arith.constant 0 : i32
        linalg.quantized_matmul ins(%arg_a, %arg_b, %0, %0 : memref<16x16xi8>, memref<16x16xi8>, i32, i32)
                                outs(%argc : memref<16x16xi32>)
        func.return
        }
        "transform.sequence"() <{"failure_propagation_mode" = 1 : i32,
                                "operandSegmentSizes" = array<i32: 0, 0>}> ({
        ^0(%arg0 : !transform.any_op, %arg1 : !transform.op<"linalg.quantized_matmul">):
            "transform.yield"() : () -> ()
        }) : () -> ()
    }
    ```
    """

    arg_types = [
        builtin.MemRefType(i8, (m, k)),  # A
        builtin.MemRefType(i8, (k, n)),  # B
        builtin.MemRefType(i32, (m, n)),  # C
    ]

    res_types = []

    @Builder.implicit_region(arg_types)
    def func_body(args: tuple[BlockArgument, ...]) -> None:
        c0 = arith.Constant.from_int_and_width(0, 32)
        linalg.QuantizedMatmulOp([args[0], args[1], c0.result, c0.result], [args[2]])
        func.Return()

    function = func.FuncOp.from_region(
        "streamer_matmul", arg_types, res_types, func_body
    )

    failurePropagationMode = builtin.IntegerAttr(1, builtin.IntegerType(32))

    input_types_t = [
        transform.AnyOpType(),
        transform.OperationType("linalg.quantized_matmul"),
    ]
    b_t = Block(arg_types=input_types_t)

    with ImplicitBuilder(b_t) as (arg0, arg1):
        (transform.TileOp(arg1, [], tiling_factors, scalable_sizes=tiling_factors))
        transform.YieldOp()

    region_t = Region(b_t)

    transform_sequence = transform.SequenceOp(failurePropagationMode, [], [], region_t)

    module = builtin.ModuleOp([function, transform_sequence])

    return module


def write_module_to_file(module, file):
    output = StringIO()
    printer = Printer(stream=output)
    printer.print(module)
    with open(file, "w") as output_file:
        output_file.write(output.getvalue())


def generate_dense_benchmark(m, n, k, add_c) -> SNAXBenchmark:
    module = create_matrix_multiply(m, n, k, add_c)
    write_module_to_file(module, "generated.transform.mlir")
    binary = "generated.x"
    bm = SNAXBenchmark(
        kernel=f"dense_matmul_{layout}_{m}x{n}x{k}",
        binary=binary,
        src_dir=str(pathlib.Path.cwd()),
        export_dir=str(pathlib.Path.cwd()),
        output_dir=str(pathlib.Path.cwd()),
    )
    return bm


if __name__ == "__main__":
    """Runs the gendata.py script with specified arguments."""

    sizes = [
        # ((128, 128, 256), (64, 64)),
        ((512, 1024, 256), (64, 64)),
        ((512, 512, 384), (32, 64)),
        ((256, 512, 512), (32, 32)),
        ((256, 256, 768), (16, 32)),
        ((128, 256, 1024), (16, 16)),
        ((128, 128, 1536), (8, 16)),
        ((64, 128, 2048), (8, 8)),
    ]

    # sizes = [
    #     ((32, 32, 32), (16, 16)),
    #     ((64, 32, 32), (16, 16)),
    # ]

    output_report: dict[str, dict] = {}

    for (
        size,
        layout,
    ) in itertools.product(sizes, ("banked",)):
        (m, n, k), tiling_factors = size

        folder = f"test_matmul_{layout}_{k}x{m}x{m}-tiled-{tiling_factors[0]}-{tiling_factors[1]}"
        bm = generate_dense_benchmark(m, n, k, tiling_factors)
        bm.clean()
        bm.build(
            build_opts=[
                "NO_CHECK=1",
                f"SIZE_M={m}",
                f"SIZE_N={n}",
                f"SIZE_K={k}",
                f"LAYOUT={layout}",
            ]
        )
        bm.run()
        bm.trace()
        bm.process_traces(folder)
        bm.copy_binary(folder)
        bm.copy_logs(folder)
        bm.copy_results()

        trace = bm.log_dir.joinpath(bm.input_file.format(hart="00000"))
        with open(trace) as file:
            data = json.load(file)
        cycles = data[1]["cycles"]

        m_tiled = tiling_factors[0]
        n_tiled = tiling_factors[1]

        size_tile = m_tiled * n_tiled * 4 + m_tiled * k + n_tiled * k
        ops_tile = m_tiled * n_tiled * k
        ai_tile = ops_tile / size_tile

        tiles = (m / m_tiled) * (n / n_tiled)

        performance = ops_tile * tiles / cycles
        avg_bandwidth = size_tile * tiles / cycles

        results = {
            "m": m,
            "n": n,
            "k": k,
            "tiling_factors": str(tiling_factors),
            "layout": layout,
            "cycles": cycles,
            "size_tile": size_tile,
            "ops_tile": ops_tile,
            "ai_tile": ai_tile,
            "tiles": tiles,
            "performance": performance,
            "avg_bandwidth": avg_bandwidth,
        }
        output_report[bm.benchmark] = results

    # Convert dictionary to pandas DataFrame
    df = pd.DataFrame.from_dict(output_report, orient="index")

    # Pretty print the DataFrame as a markdown table
    assert (markdown_table := df.to_markdown()) is not None

    with open("output/index.md", "w") as file:
        file.write(markdown_table)

