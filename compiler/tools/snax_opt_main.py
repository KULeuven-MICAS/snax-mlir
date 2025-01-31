import argparse
from collections.abc import Sequence

from xdsl.context import MLContext
from xdsl.dialects import get_all_dialects
from xdsl.transforms import get_all_passes
from xdsl.xdsl_opt_main import xDSLOptMain

from compiler.dialects import get_all_snax_dialects
from compiler.transforms import get_all_snax_passes


class SNAXOptMain(xDSLOptMain):
    def register_all_dialects(self):
        all_dialects = get_all_dialects()
        # FIXME: override upstream accfg and stream dialect.
        all_dialects.pop("accfg", None)
        all_dialects.pop("stream", None)
        all_dialects.update(get_all_snax_dialects())
        for dialect_name, dialect_factory in all_dialects.items():
            self.ctx.register_dialect(dialect_name, dialect_factory)

    def register_all_passes(self):
        """
        Register all SNAX and xDSL passes
        """
        all_passes = get_all_passes()
        all_passes.update(get_all_snax_passes())
        for pass_name, pass_factory in all_passes.items():
            self.register_pass(pass_name, pass_factory)

    def __init__(
        self,
        description: str = "SNAX modular optimizer driver",
        args: Sequence[str] | None = None,
    ):
        self.available_frontends = {}
        self.available_passes = {}
        self.available_targets = {}

        self.ctx = MLContext()
        self.register_all_dialects()
        super().register_all_frontends()
        self.register_all_passes()
        super().register_all_targets()

        # arg handling
        arg_parser = argparse.ArgumentParser(description=description)
        super().register_all_arguments(arg_parser)
        self.args = arg_parser.parse_args(args=args)

        self.ctx.allow_unregistered = self.args.allow_unregistered_dialect

        super().setup_pipeline()


def main():
    SNAXOptMain().run()


if "__main__" == __name__:
    main()
