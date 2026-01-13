import argparse
import subprocess
import sys
from collections.abc import Sequence
from io import StringIO

from xdsl.dialects.builtin import ModuleOp
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PassPipeline
from xdsl.printer import Printer
from xdsl.transforms.mlir_opt import MLIROptPass

from snaxc.accelerators.acc_context import AccContext
from snaxc.tools.snaxc_main import SNAXCMain
from snaxc.transforms.phs.convert_pe_to_hw import ConvertPEToHWPass
from snaxc.transforms.phs.encode import PhsEncodePass
from snaxc.transforms.phs.export_phs import PhsKeepPhsPass, PhsRemovePhsPass
from snaxc.transforms.phs.finalize_phs_to_hw import FinalizePhsToHWPass


class PHSCMain(SNAXCMain):
    def __init__(
        self,
        description: str = "Programmable Hardware Synthesis Compiler",
        args: Sequence[str] | None = None,
    ):
        # arg handling
        arg_parser = argparse.ArgumentParser(description=description)
        self.register_all_arguments(arg_parser)
        self.args = arg_parser.parse_args(args=args)
        self.load_config()
        # self.ctx = AccContext(allow_unregistered=True)

        self.register_all_dialects()
        self.setup_pipelines()

    def run(self):
        # read file
        f = open(self.args.input_file)
        module = Parser(self.ctx, f.read(), self.get_input_name()).parse_module()
        f.close()

        # apply passes
        module.verify()
        self.input_pipeline.apply(self.ctx, module)
        module.verify()
        hardware_module = module.clone()

        self.hardware_pipeline.apply(self.ctx, hardware_module)
        hardware_module.verify()

        # If an optional explicit software file is requested, overwrite the previous module
        if self.args.software_file:
            f = open(self.args.software_file)
            module = Parser(self.ctx, f.read(), self.args.software_file).parse_module()
            f.close()

        self.software_pipeline.apply(self.ctx, module)
        module.verify()

        # write to output
        output_hardware_stream = StringIO()
        Printer(output_hardware_stream).print_op(hardware_module)
        hardware_ir_string = output_hardware_stream.getvalue()

        # Hardware postprocessing pipeline treats circt-opt and firtool as black box
        # Because the output after circt-opt can not be parsed by xdsl,
        # and for sure the systemverilog after firtool can not be parsed by xdsl.

        if not self.args.no_sv_conversion:
            p1 = subprocess.Popen(
                ["circt-opt", "--map-arith-to-comb", "--hw-flatten-modules"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            p2 = subprocess.Popen(
                ["firtool", "--format=mlir", "--strip-debug-info"],
                stdin=p1.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            assert p1.stdout is not None
            p1.stdout.close()
            _, p1_stderr = p1.communicate(input=hardware_ir_string)
            if p1.returncode != 0:
                print(
                    f"Error during hardware conversion (circt-opt):\n{p1_stderr}",
                    file=sys.stderr,
                )
                raise SystemExit(p1.returncode or 1)
            stdout_final, stderr_final = p2.communicate()
            if p2.returncode != 0:
                print(
                    f"Error during hardware conversion (firtool):\n{stderr_final}",
                    file=sys.stderr,
                )
                raise SystemExit(p2.returncode or 1)
            else:
                with open(self.args.output_hardware, "w") as outfile:
                    outfile.write(stdout_final)

        else:
            with open(self.args.output_hardware, "w") as outfile:
                outfile.write(hardware_ir_string)

        output_software_stream = open(self.args.output_file, "w")
        Printer(output_software_stream).print_op(module)
        output_software_stream.write("\n")
        output_software_stream.flush()

        # Go to

        if output_software_stream is not sys.stdout:
            output_software_stream.close()

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        """
        Registers all the command line arguments that are used by this tool.

        Add other/additional arguments by overloading this function.
        """

        super().register_all_arguments(arg_parser)

        arg_parser.add_argument("schedule_file", type=str, nargs="?", help="path to schedule file")
        arg_parser.add_argument(
            "--software-file",
            type=str,
            nargs="?",
            help="path to separate other software stream,"
            " by default the same input stream is used for hard- and software",
        )
        arg_parser.add_argument("--output-hardware", type=str, required=True, help="path to output hardware")
        arg_parser.add_argument(
            "--no-sv-conversion", action="store_true", help="Don't convert output hardware to systemverilog"
        )

    def setup_pipelines(self):
        """
        Creates multiple pipelines that consists of all the passes specified:

        ```
        input_file,
        schedule_file,
          V
          | <- input pipeline
          *
          |\\
          | \\
          | | <- hardware pipeline
          | |
          | x acc_array.mlir
          | |
          | | <- hardware postprocessing pipeline
          | |
          | x acc_array.sv
          |
          | <- software pipeline
          |
          x input_file_preprocessed.mlir
        ```

        Fails, if not all passes are registered.
        """

        assert isinstance(self.ctx, AccContext), "Context must be an AccContext instance"

        def set_input_pipeline():
            """
            Create input pipeline.
            The input pipeline annotates and encodes relevant linalg ops into PHS
            """
            input_pass_pipeline: list[ModulePass] = []

            input_pass_pipeline.append(
                MLIROptPass(
                    arguments=(
                        "--linalg-generalize-named-ops",
                        f"--transform-preload-library=transform-library-paths={self.args.schedule_file}",
                        "--transform-interpreter",
                    )
                )
            )
            input_pass_pipeline.append(PhsEncodePass())
            return input_pass_pipeline

        def set_hardware_pipeline():
            hardware_pass_pipeline: list[ModulePass] = []
            hardware_pass_pipeline.append(PhsKeepPhsPass())
            hardware_pass_pipeline.append(ConvertPEToHWPass((4,)))
            hardware_pass_pipeline.append(FinalizePhsToHWPass())
            return hardware_pass_pipeline

        snaxc_pipeline_setup = super().setup_pipeline

        def set_software_pipeline():
            software_pass_pipeline: list[ModulePass] = []
            software_pass_pipeline.append(PhsRemovePhsPass())

            # Get the normal pipeline from SNAXC
            snaxc_pipeline_setup()
            software_pass_pipeline.extend(self.pipeline.passes)
            delattr(self, "pipeline")
            return software_pass_pipeline

        def callback(previous_pass: ModulePass, module: ModuleOp, next_pass: ModulePass) -> None:
            module.verify()
            if self.args.print_between_passes:
                print(f"// IR after {previous_pass.name}:")
                printer = Printer(stream=sys.stdout)
                printer.print_op(module)
                print("\n\n\n")

        # Initialize pipelines
        self.input_pipeline = PassPipeline(tuple(set_input_pipeline()), callback)
        self.hardware_pipeline = PassPipeline(tuple(set_hardware_pipeline()), callback)
        self.software_pipeline = PassPipeline(tuple(set_software_pipeline()), callback)


def main():
    PHSCMain().run()


if "__main__" == __name__:
    main()
