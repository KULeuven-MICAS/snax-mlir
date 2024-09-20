import pathlib
from io import StringIO

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, builtin, func, linalg, transform
from xdsl.dialects.builtin import i8
from xdsl.ir import Block, Region
from xdsl.printer import Printer

from util.snax_benchmark import SNAXBenchmark


def create_tiled_matrix_multiply(k, m, n, tiling_factors):
    """
    Generate IR in the form of:
    ```
    builtin.module {
      func.func @streamer_matmul(%arg0 : memref<16x16xi8>, %arg1 : memref<16x16xi8,
                                 strided<[1, 16]>>, %arg2 : memref<16x16xi32>) {
        %0 = arith.constant 0 : i32
        linalg.quantized_matmul ins(%arg0, %arg1, %0, %0 : memref<16x16xi8>,
                                    memref<16x16xi8, strided<[1, 16]>>, i32, i32)
                                outs(%arg2 : memref<16x16xi32>)
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

    def get_2d_memref_type(typ, dim_one, dim_two, transpose=False):
        layout = (
            builtin.StridedLayoutAttr([1, dim_one]) if transpose else builtin.NoneAttr()
        )
        return builtin.MemRefType(typ, [dim_one, dim_two], layout=layout)

    input_types = [
        get_2d_memref_type(i8, k, m),
        get_2d_memref_type(i8, m, n, transpose=True),
        get_2d_memref_type(builtin.i32, k, n),
    ]

    b = Block(arg_types=(input_types))

    with ImplicitBuilder(b) as (arg0, arg1, arg2):
        c0 = arith.Constant.from_int_and_width(0, 32)
        linalg.QuantizedMatmulOp([arg0, arg1, c0.result, c0.result], [arg2])
        func.Return()

    region = Region(b)

    function = func.FuncOp.from_region("streamer_matmul", input_types, [], region)

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


def generate_tiled_benchmark(m, n, k, tiling_factors) -> SNAXBenchmark:
    module = create_tiled_matrix_multiply(k, m, n, tiling_factors)
    write_module_to_file(module, "generated.transform.mlir")
    binary = "generated.stream.x"
    bm = SNAXBenchmark(
        kernel=f"tiled_matmul_generated_{k}x{n}x{m}",
        binary=binary,
        src_dir=str(pathlib.Path.cwd()),
        export_dir=str(pathlib.Path.cwd()),
    )
    return bm


if __name__ == "__main__":
    """Runs the gendata.py script with specified arguments."""
    sizes = [
        [16, 16, 16],
        [32, 32, 32],
        [64, 64, 64],
        [128, 128, 128],
        [256, 256, 256],
        [512, 512, 512],
    ]
    options = ["NO_ACCFG_OPT=1", "DEDUP_ONLY=1", "OVERLAP_ONLY=1", "ACCFG_BOTH=1"]
    combined_list = [[size, option] for size in sizes for option in options]
    for size, options in combined_list:
        k, m, n = size
        tiling_factors = [8, 8]
        folder = f"test_generated_{k}x{m}x{m}_{options[:-2]}"
        bm = generate_tiled_benchmark(k, m, n, tiling_factors)
        bm.clean()
        bm.build(
            build_opts=[
                "NO_CHECK=1",
                options,
                f"SIZE_M={m}",
                f"SIZE_N={n}",
                f"SIZE_K={k}",
            ]
        )
        bm.run()
        bm.trace()
        bm.process_traces(folder)
        bm.copy_binary(folder)
        bm.copy_logs(folder)


#    def run_all(binary: str, folder: str):
#        bm = SNAXBenchmark(
#            kernel="streamer_matmul",
#            binary=binary,
#            src_dir=str(pathlib.Path.cwd()),
#            export_dir=str(pathlib.Path.cwd()),
#        )
#        bm.clean()
#        bm.build(build_opts=[])
#        bm.run()
#        bm.trace()
#        bm.process_traces(folder)
#        bm.copy_binary(folder)
#        bm.copy_logs(folder)
#
#    binaries = {
#        "run0": "matmul.x",
#        "run1": "half_tiled_matmul.x",
#        "run2": "transform_matmul.x",
#        "run3": "dynamic_matmul.x",
#    }
#
#    for folder, binary in binaries.items():
#        run_all(binary, folder)
