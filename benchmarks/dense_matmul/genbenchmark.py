import pathlib
from io import StringIO

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, builtin, func, linalg, transform
from xdsl.dialects.builtin import i8, i32
from xdsl.ir import Block, Region
from xdsl.printer import Printer

from util.snax_benchmark import SNAXBenchmark


def create_tiled_matrix_multiply(k, m, n):
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
    }
    ```
    """

    def get_2d_memref_type(typ, dim_one, dim_two, transpose=False):
        layout = builtin.StridedLayoutAttr([1, dim_one]) if transpose else builtin.NoneAttr()
        return builtin.MemRefType(typ, [dim_one, dim_two], layout=layout)

    input_types = [
        get_2d_memref_type(i8, k, m),
        get_2d_memref_type(i8, m, n, transpose=True),
        get_2d_memref_type(i32, k, n),
    ]

    b = Block(arg_types=(input_types))

    with ImplicitBuilder(b) as (arg0, arg1, arg2):
        c0 = arith.Constant.from_int_and_width(0, 32)
        linalg.QuantizedMatmulOp([arg0, arg1, c0.result, c0.result], [arg2])
        func.Return()

    region = Region(b)

    function = func.FuncOp.from_region("streamer_matmul", input_types, [], region)

    module = builtin.ModuleOp([function])

    return module


def write_module_to_file(module, file):
    output = StringIO()
    printer = Printer(stream=output)
    printer.print(module)
    with open(file, "w") as output_file:
        output_file.write(output.getvalue())


def generate_tiled_benchmark(m, n, k) -> SNAXBenchmark:
    module = create_tiled_matrix_multiply(k, m, n)
    write_module_to_file(module, "generated.mlir")
    binary = "generated.x"
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
#         [64, 64, 64],
#         [128, 128, 128],
#         [256, 256, 256],
#         [512, 512, 512],
    ]
    for size in sizes:
        k, m, n = size
        folder = f"test_generated_{k}x{m}x{m}"
        bm = generate_tiled_benchmark(k, m, n)
        bm.clean()
        bm.build(
            build_opts=[
                "NO_CHECK=1",
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
