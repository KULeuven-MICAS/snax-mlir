import itertools
import json
import pathlib
from io import StringIO

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, builtin, func, linalg
from xdsl.dialects.builtin import i8, i32
from xdsl.ir import Block, Region
from xdsl.printer import Printer

from util.snax_benchmark import SNAXBenchmark


def create_matrix_multiply(k, m, n):
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
        layout = (
            builtin.StridedLayoutAttr([1, dim_one]) if transpose else builtin.NoneAttr()
        )
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
    module = create_matrix_multiply(k, m, n)
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
    selected_dims = [32, 48, 64]

    sizes = list(itertools.product(selected_dims, repeat=3))

    output_report: dict[str, dict] = {}

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

        # add to output report
        trace = bm.log_dir.joinpath(bm.input_file.format(hart="00000"))
        with open(trace) as file:
            data = json.load(file)
        cycles = data[1]["cycles"]
        ideal = round((k / 8 + 1) * (m / 8) * (n / 8))
        utilization = ideal / cycles
        output_report[bm.benchmark] = {
            "cycles": cycles,
            "ideal": ideal,
            "utilization": utilization,
        }

    with open("output_report.txt", "w") as file:
        file.write("benchmark\tcycles\tideal\tutilization\t\n")
        avg_utilization = 0
        avg_n = 0
        for benchmark in output_report:
            file.write(f"{benchmark}\t")
            file.write(f"{output_report[benchmark]['cycles']}\t")
            file.write(f"{output_report[benchmark]['ideal']}\t")
            file.write(f"{output_report[benchmark]['utilization']}\t")
            file.write("\n")
            avg_utilization += output_report[benchmark]["utilization"]
            avg_n += 1
        file.write("--------------------------\n")
        file.write("average\t\t\t")
        file.write(f"{avg_utilization/avg_n}\t\n")
