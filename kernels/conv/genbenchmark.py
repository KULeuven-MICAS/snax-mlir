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

def write_makefile(file):

    with open(file, "w") as output_file:
        output_file.write("include $(realpath ../../Makefile)\n")
        output_file.write("clean:\n")
        output_file.write("\trm -rf logs/*.txt logs/*.dasm\n")

def generate_temp_mapping_benchmark():

    spec = ConvSpec(1, 16, 16, 3, 3, 16, 16)

    module = generate_conv_ir(spec)

    for schedule_idx in range(1440):

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
        bm.build([f"SCHEDULE_IDX={schedule_idx}", "PURE_OUTPUT_SATIONARY=false"])
        #bm.run()
        #bm.trace()
        bm.copy_binary('')
        write_makefile(bm.export_dir / 'Makefile')
        #bm.copy_logs('')
        bm.clean()

def generate_resnet_benchmark():

    specs = [
        # input layer:
        ConvSpec(1, 112, 112, 7, 7, 3, 64, stride=2),

        # block 1:
        # bottleneck:
        ConvSpec(1, 56, 56, 1, 1, 256, 64),
        ConvSpec(1, 56, 56, 3, 3, 64, 64),
        ConvSpec(1, 56, 56, 1, 1, 64, 256),

        # block 2:
        # downsampling layer:
        ConvSpec(1, 28, 28, 1, 1, 256, 512, stride=2),

        # bottleneck:
        ConvSpec(1, 28, 28, 1, 1, 256, 128),
        ConvSpec(1, 28, 28, 1, 1, 512, 128),
        ConvSpec(1, 28, 28, 3, 3, 128, 128),
        ConvSpec(1, 28, 28, 1, 1, 128, 512),

        # block 3:
        # downsampling layer:
        ConvSpec(1, 14, 14, 1, 1, 512, 1024, stride=2),

        # bottleneck:
        ConvSpec(1, 14, 14, 1, 1, 512, 256),
        ConvSpec(1, 14, 14, 1, 1, 1024, 256),
        ConvSpec(1, 14, 14, 3, 3, 256, 256),
        ConvSpec(1, 14, 14, 1, 1, 256, 1024),

        # block 4:
        # downsampling layer:
        ConvSpec(1, 7, 7, 1, 1, 1024, 512, stride=2),

        # bottleneck:
        ConvSpec(1, 7, 7, 1, 1, 1024, 512),
        ConvSpec(1, 7, 7, 1, 1, 2048, 512),
        ConvSpec(1, 7, 7, 3, 3, 512, 512),
        ConvSpec(1, 7, 7, 1, 1, 512, 2048),
    ]


    for i, spec in enumerate(specs[0:1]):

        module = generate_conv_ir(spec, generate_constants = False)

        binary = "generated.x"
        bm = SNAXBenchmark(
            kernel=f"resnet50_{i:03d}",
            binary=binary,
            src_dir=str(pathlib.Path.cwd()),
            export_dir=str(pathlib.Path.cwd()),
            output_dir=str(pathlib.Path.cwd()),
        )

        bm.clean()
        write_module_to_file(module, "generated.mlir")
        bm.build(["PURE_OUTPUT_SATIONARY=true"])
        #bm.run()
        #bm.trace()
        bm.copy_binary('')
        write_makefile(bm.export_dir / 'Makefile')
        #bm.copy_logs('')
        bm.clean()

if __name__ == "__main__":

    # generate_temp_mapping_benchmark()
    generate_resnet_benchmark()
