import itertools
import json
import pathlib
from datetime import datetime
from io import StringIO

from xdsl.builder import Builder
from xdsl.dialects import arith, builtin, func, linalg, tensor
from xdsl.dialects.builtin import i8, i32
from xdsl.ir import BlockArgument
from xdsl.printer import Printer

from util.snax_benchmark import SNAXBenchmark


def create_matrix_multiply(m, n, k, add_c: bool = False):
    """
    Generate IR for a matmul / gemm in the form of:
    ```
    builtin.module {
        func.func @streamer_matmul(
            %arg_a : tensor<16x16xi8>,
            %arg_b : tensor<16x16xi32>,
            %arg_c : tensor<16x16xi32>) -> tensor<16x16xi32>{
        %0 = arith.constant 0 : i32
        %1 = linalg.quantized_matmul ins(%arg_a, %arg_b, %0, %0 : tensor<16x16xi8>, tensor<16x16xi8>, i32, i32)
                                outs(%arg2 : tensor<16x16xi32>)
        (optional) := %2 = linalg.add %1, %arg_c ...
        func.return %1 / %2
        }
    }
    ```
    """

    arg_types = [
        builtin.TensorType(i8, (m, k)),  # A
        builtin.TensorType(i8, (k, n)),  # B
        builtin.TensorType(i32, (m, n)),  # C
    ]

    res_types = [
        builtin.TensorType(i32, (m, n)),  # D
    ]

    @Builder.implicit_region(arg_types)
    def func_body(args: tuple[BlockArgument, ...]) -> None:
        c0 = arith.ConstantOp.from_int_and_width(0, 32)
        empty_tensor = tensor.EmptyOp([], (arg_types[-1]))
        result = linalg.QuantizedMatmulOp(
            [args[0], args[1], c0.result, c0.result], [empty_tensor.tensor]
        )
        if add_c:
            empty_tensor_2 = tensor.EmptyOp([], (arg_types[-1]))
            newresult = linalg.AddOp(
                [args[2], result.results[0]], [empty_tensor_2.tensor]
            )
            func.ReturnOp(newresult)
        else:
            func.ReturnOp(result)

    function = func.FuncOp.from_region(
        "streamer_matmul", arg_types, res_types, func_body
    )

    module = builtin.ModuleOp([function])

    return module


def write_module_to_file(module, file):
    output = StringIO()
    printer = Printer(stream=output)
    printer.print(module)
    with open(file, "w") as output_file:
        output_file.write(output.getvalue())


def generate_dense_benchmark(m, n, k, add_c) -> SNAXBenchmark:
    module = create_matrix_multiply(m, n, k, add_c)
    write_module_to_file(module, "generated.mlir")
    binary = "generated.x"
    bm = SNAXBenchmark(
        kernel=f"dense_{'gemm' if add_c else 'matmul'}_{layout}_{m}x{n}x{k}",
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
    for layout, add_c in itertools.product(("cyclic",), (True, False)):
        result += f"Results for a {layout} layout {'with add C' if add_c else ''} \n\n"
        result += "| benchmark | layout | add C | M | N | K | plots | cycles | ideal | utilization |\n"
        result += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        avg_utilization = 0
        avg_n = 0
        for benchmark in output_report:
            if output_report[benchmark]["layout"] != layout:
                continue
            if output_report[benchmark]["add_c"] != add_c:
                continue
            result += f"| [{benchmark}]({benchmark}) "
            result += f"| {output_report[benchmark]['layout']} "
            result += f"| {'yes' if output_report[benchmark]['add_c'] else 'no'} "
            result += f"| {output_report[benchmark]['m']} "
            result += f"| {output_report[benchmark]['n']} "
            result += f"| {output_report[benchmark]['k']} "
            result += (
                f"| {'yes' if output_report[benchmark]['plots_available'] else 'no'} "
            )
            result += f"| {output_report[benchmark]['cycles']} "
            result += f"| {output_report[benchmark]['ideal']} "
            result += f"| {output_report[benchmark]['utilization']} | \n"
            avg_utilization += output_report[benchmark]["utilization"]
            avg_n += 1
        result += "| average | | | | | | | |"
        result += f"{avg_utilization/avg_n} |\n\n"
    return result


def output_log_benchmark(
    benchmark_name: str, utilization: dict[str, int], to_plot: bool
) -> str:
    result: str = ""
    result += f"# results for {benchmark_name}\n\n"
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    result += f"This test was run at {dt_string}\n\n"
    result += f"Utilization: {utilization['utilization']}\n\n"
    result += f" ({utilization['ideal']} cycles ideal, {utilization['cycles']} cycles real)\n\n"
    if to_plot:
        result += "[view banking conflicts plot](figures/banking_conflicts.pdf)\n\n"
    result += f"[dowload logs and binaries that generated this result]({benchmark_name}_results.tar.gz)\n\n"
    if to_plot:
        result += "![conflicts_bank](figures/nb_of_stalls_per_bank.png)\n\n"
        result += "![conflicts_port](figures/nb_of_stalls_per_port.png)\n\n"
    return result


if __name__ == "__main__":
    """Runs the gendata.py script with specified arguments."""
    selected_dims = [32, 48, 64]

    sizes = list(itertools.product(selected_dims, repeat=3))

    output_report: dict[str, dict] = {}

    for size, layout, add_c in itertools.product(sizes, ("cyclic",), (True, False)):
        m, n, k = size

        # plot:
        # only plot if max(m,n,k) <= 48
        to_plot = max(m, n, k) <= 48
        folder = f"test_{'gemm' if add_c else 'matmul'}_{layout}_{k}x{m}x{m}"
        bm = generate_dense_benchmark(m, n, k, add_c)
        bm.clean()
        bm.build(
            build_opts=[
                "NO_CHECK=1",
                f"SIZE_M={m}",
                f"SIZE_N={n}",
                f"SIZE_K={k}",
                f"LAYOUT={layout}",
                f"ADD_C={int(add_c)}",
            ]
        )
        bm.run()
        bm.trace()
        if to_plot:
            bm.plot()
        bm.process_traces(folder, accelerator="snax_gemmx")
        bm.copy_binary(folder)
        bm.copy_logs(folder)
        if to_plot:
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
            "add_c": add_c,
            "cycles": cycles,
            "ideal": ideal,
            "utilization": utilization,
            "plots_available": to_plot,
        }
        output_report[bm.benchmark] = results

        bm.generate_output_log(
            lambda name: output_log_benchmark(name, results, to_plot)
        )

    with open("output/index.md", "w") as file:
        file.write(output_log(output_report))
