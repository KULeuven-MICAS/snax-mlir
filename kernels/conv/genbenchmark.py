import pathlib

from genkernel import ConvSpec, generate_conv_ir
from xdsl.ir import StringIO
from xdsl.printer import Printer

from util.snax_benchmark import SNAXBenchmark


def write_module_to_file(module, file):
    output = StringIO()
    printer = Printer(stream=output)
    printer.print(module)
    with open(file, "w") as output_file:
        output_file.write(output.getvalue())


if __name__ == "__main__":

    spec = ConvSpec(1, 16, 16, 3, 3, 16, 16)
    module = generate_conv_ir(spec)

    for schedule_idx in range(72):

        binary = "generated.x"
        bm = SNAXBenchmark(
            kernel=f"conv_{schedule_idx:04d}",
            binary=binary,
            src_dir=str(pathlib.Path.cwd()),
            export_dir=str(pathlib.Path.cwd()),
            output_dir=str(pathlib.Path.cwd()),
        )

        bm.clean()
        write_module_to_file(module, "generated.mlir")
        bm.build([f"SCHEDULE_IDX={schedule_idx}"])
        #bm.run()
        #bm.trace()
        bm.copy_binary('')
        #bm.copy_logs('')
        bm.clean()
