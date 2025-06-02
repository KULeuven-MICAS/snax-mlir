import argparse
import sys
from collections.abc import Sequence

from xdsl.context import Context
from xdsl.dialects import get_all_dialects
from xdsl.dialects.builtin import ModuleOp
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PipelinePass
from xdsl.printer import Printer
from xdsl.tools.command_line_tool import CommandLineTool

from snaxc.dialects import get_all_snax_dialects


class SNAXCMain(CommandLineTool):
    def __init__(
        self,
        description: str = "SNAX-MLIR Compiler",
        args: Sequence[str] | None = None,
    ):
        self.ctx = Context(allow_unregistered=True)
        self.register_all_dialects()

        # arg handling
        arg_parser = argparse.ArgumentParser(description=description)
        self.register_all_arguments(arg_parser)
        self.args = arg_parser.parse_args(args=args)

        self.setup_pipeline()

    def register_all_dialects(self):
        all_dialects = get_all_dialects()
        # FIXME: override upstream accfg and stream dialect.
        all_dialects.pop("accfg", None)
        all_dialects.pop("stream", None)
        all_dialects.update(get_all_snax_dialects())
        for dialect_name, dialect_factory in all_dialects.items():
            self.ctx.register_dialect(dialect_name, dialect_factory)

    def run(self):
        # read file
        f = open(self.args.input_file)
        module = Parser(self.ctx, f.read(), self.get_input_name()).parse_module()
        f.close()

        # apply passes
        module.verify()
        self.pipeline.apply(self.ctx, module)
        module.verify()

        # write to output
        output_stream = open(self.args.output_file, "w")
        Printer(output_stream).print_op(module)
        output_stream.write("\n")
        output_stream.flush()

        if output_stream is not sys.stdout:
            output_stream.close()

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        """
        Registers all the command line arguments that are used by this tool.

        Add other/additional arguments by overloading this function.
        """

        arg_parser.add_argument(
            "input_file", type=str, nargs="?", help="path to input file"
        )

        arg_parser.add_argument(
            "-o", "--output-file", type=str, required=True, help="path to output file"
        )

        arg_parser.add_argument(
            "--print-between-passes",
            default=False,
            action="store_true",
            help="Print the IR between each pass",
        )

        arg_parser.add_argument(
            "--print-op-generic",
            default=False,
            action="store_true",
            help="Print operations with the generic format",
        )

    def setup_pipeline(self):
        """
        Creates a pipeline that consists of all the passes specified.

        Fails, if not all passes are registered.
        """

        def callback(
            previous_pass: ModulePass, module: ModuleOp, next_pass: ModulePass
        ) -> None:
            module.verify()
            if self.args.print_between_passes:
                print(f"IR after {previous_pass.name}:")
                printer = Printer(stream=sys.stdout)
                printer.print_op(module)
                print("\n\n\n")

        self.pipeline = PipelinePass(tuple(), callback)


def main():
    SNAXCMain().run()


if "__main__" == __name__:
    main()
