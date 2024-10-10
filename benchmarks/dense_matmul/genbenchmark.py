import itertools
import json
import pathlib
from datetime import datetime
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


def generate_dense_benchmark(m, n, k) -> SNAXBenchmark:
    module = create_matrix_multiply(k, m, n)
    write_module_to_file(module, "generated.mlir")
    binary = "generated.x"
    bm = SNAXBenchmark(
        kernel=f"dense_matmul_{layout}_{k}x{n}x{m}",
        binary=binary,
        src_dir=str(pathlib.Path.cwd()),
        export_dir=str(pathlib.Path.cwd()),
        output_dir=str(pathlib.Path.cwd()),
    )
    return bm


def output_log(output_report) -> str:
    result = "# Dense Matmul Benchmark Results\n\n"
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    result += f"This test was run at {dt_string}\n\n"
    for layout in ("cyclic", "banked"):
        result += f"Results for a {layout} layout \n\n"
        result += "| benchmark | layout | M | N | K | cycles | ideal | utilization |\n"
        result += "| --- | --- | --- | --- |\n"
        avg_utilization = 0
        avg_n = 0
        for benchmark in output_report:
            if output_report[benchmark]["layout"] != layout:
                continue
            result += f"| [{benchmark}]({benchmark}) "
            result += f"| {output_report[benchmark]['layout']} "
            result += f"| {output_report[benchmark]['m']} "
            result += f"| {output_report[benchmark]['n']} "
            result += f"| {output_report[benchmark]['k']} "
            result += f"| {output_report[benchmark]['cycles']} "
            result += f"| {output_report[benchmark]['ideal']} "
            result += f"| {output_report[benchmark]['utilization']} | \n"
            avg_utilization += output_report[benchmark]["utilization"]
            avg_n += 1
        result += "| average | | |"
        result += f"{avg_utilization/avg_n} |\n\n"
    return result


def output_log_benchmark(benchmark_name: str, utilization: dict[str, int]) -> str:
    result: str = ""
    result += f"# results for {benchmark_name}\n\n"
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    result += f"This test was run at {dt_string}\n\n"
    result += f"Utilization: {utilization['utilization']}\n\n"
    result += f" ({utilization['ideal']} cycles ideal, {utilization['cycles']} cycles real)\n\n"
    result += "[view banking conflicts plot](figures/banking_conflicts.pdf)\n\n"
    result += f"[dowload logs and binaries that generated this result]({benchmark_name}_results.tar.gz)\n\n"
    result += "![conflicts_bank](figures/nb_of_stalls_per_bank.png)\n\n"
    result += "![conflicts_port](figures/nb_of_stalls_per_port.png)\n\n"
    return result


if __name__ == "__main__":
    """Runs the gendata.py script with specified arguments."""
    selected_dims = [32, 48, 64]

    sizes = list(itertools.product(selected_dims, repeat=3))

    nn_size = [
        {"M": 16, "K": 512, "N": 32},
        {"M": 448, "K": 32, "N": 32},
        {"M": 8, "K": 32, "N": 192},
        {"M": 8, "K": 16, "N": 16},
        {"M": 224, "K": 192, "N": 16},
        {"M": 8, "K": 16, "N": 96},
        {"M": 8, "K": 16, "N": 96},
        {"M": 64, "K": 96, "N": 24},
        {"M": 8, "K": 24, "N": 48},
        {"M": 56, "K": 16, "N": 48},
        {"M": 8, "K": 144, "N": 32},
        {"M": 56, "K": 32, "N": 32},
        {"M": 200, "K": 16, "N": 48},
        {"M": 200, "K": 192, "N": 64},
        {"M": 200, "K": 64, "N": 32},
        {"M": 200, "K": 16, "N": 96},
        {"M": 200, "K": 384, "N": 8},
        {"M": 200, "K": 96, "N": 8},
        {"M": 56, "K": 16, "N": 576},
        {"M": 8, "K": 576, "N": 160},
        {"M": 56, "K": 160, "N": 48},
        {"M": 8, "K": 16, "N": 960},
        {"M": 56, "K": 960, "N": 64},
        {"M": 56, "K": 320, "N": 64},
        {"M": 8, "K": 1280, "N": 40},
        {"M": 8, "K": 152, "N": 32},
        {"M": 8, "K": 576, "N": 64},
        {"M": 8, "K": 576, "N": 128},
        {"M": 112, "K": 128, "N": 128},
        {"M": 56, "K": 64, "N": 32},
        {"M": 40, "K": 1152, "N": 64},
        {"M": 200, "K": 192, "N": 64},
        {"M": 200, "K": 128, "N": 32},
        {"M": 56, "K": 576, "N": 8},
        {"M": 56, "K": 512, "N": 8},
        {"M": 56, "K": 256, "N": 128},
        {"M": 8, "K": 512, "N": 200},
        {"M": 40, "K": 768, "N": 8},
        {"M": 40, "K": 768, "N": 96},
        {"M": 40, "K": 64, "N": 200},
        {"M": 200, "K": 200, "N": 64},
        {"M": 40, "K": 768, "N": 8},
        {"M": 40, "K": 768, "N": 8},
        {"M": 8, "K": 192, "N": 128},
        {"M": 8, "K": 768, "N": 40},
        {"M": 32, "K": 768, "N": 64},
        {"M": 8, "K": 64, "N": 512},
        {"M": 32, "K": 512, "N": 64},
        {"M": 128, "K": 768, "N": 8},
        {"M": 128, "K": 792, "N": 8},
        {"M": 128, "K": 192, "N": 88},
    ]
    sizes += nn_size

    output_report: dict[str, dict] = {}

    for size, layout in itertools.product(sizes, ("cyclic", "banked")):
        k, m, n = size
        folder = f"test_{layout}_{k}x{m}x{m}"
        bm = generate_dense_benchmark(k, m, n)
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
        bm.plot()
        bm.process_traces(folder)
        bm.copy_binary(folder)
        bm.copy_logs(folder)
        bm.copy_plots()
        bm.copy_results()

        # add to output report
        trace = bm.log_dir.joinpath(bm.input_file.format(hart="00000"))
        with open(trace) as file:
            data = json.load(file)
        cycles = data[1]["cycles"]
        ideal = round((k / 8 + 1) * (m / 8) * (n / 8))
        utilization = ideal / cycles
        results = {
            "m": m,
            "n": n,
            "k": k,
            "layout": layout,
            "cycles": cycles,
            "ideal": ideal,
            "utilization": utilization,
        }
        output_report[bm.benchmark] = results

        bm.generate_output_log(lambda name: output_log_benchmark(name, results))

    with open("output/index.md", "w") as file:
        file.write(output_log(output_report))
