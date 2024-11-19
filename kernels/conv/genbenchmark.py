import pathlib
import shutil

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

def write_makefile(file, schedule_idx = None):

    with open(file, "w") as output_file:
        output_file.write("include $(realpath ../../Makefile)\n")
        if schedule_idx is not None:
            output_file.write(f'SCHEDULE_IDX={schedule_idx}\n')
        output_file.write("clean:\n")
        output_file.write("\trm -rf logs/*.txt logs/*.dasm\n")

def generate_temp_mapping_benchmark():

    # conv 3x3
    # spec = ConvSpec(1, 32, 16, 3, 3, 64, 64)

    # conv 7x7
    spec = ConvSpec(1, 16, 16, 7, 7, 64, 16)

    # pw conv
    # spec = ConvSpec(1, 14, 14, 1, 1, 1024, 256)

    # gemm
    # spec = ConvSpec(1, 256, 1, 1, 1, 256, 256)

    module = generate_conv_ir(spec, generate_constants=False)

    for schedule_idx in range(720):

        binary = "generated.x"
        bm = SNAXBenchmark(
            kernel=f"conv_{schedule_idx:04d}",
            binary=binary,
            src_dir=str(pathlib.Path.cwd()),
            export_dir=str(pathlib.Path.cwd()),
            output_dir=str(pathlib.Path.cwd()),
        )

        # bm.cdst_folder = pathlib.Path(self.export_dir / folder)
        if not bm.export_dir.exists():
            bm.export_dir.mkdir(parents=True)
        # write_module_to_file(module, bm.export_dir / "generated.mlir")
        write_module_to_file(module, bm.export_dir / "generated.mlir")
        # bm.build([f"SCHEDULE_IDX={schedule_idx}", "PURE_OUTPUT_SATIONARY=false"])
        #bm.run()
        #bm.trace()
        shutil.copy(src=bm.src_dir / 'main.c', dst=bm.export_dir / 'main.c')
        write_makefile(bm.export_dir / 'Makefile', schedule_idx)
        #bm.copy_logs('')
        # bm.clean()

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


    for i, spec in enumerate(specs):

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

    generate_temp_mapping_benchmark()
    # generate_resnet_benchmark()
