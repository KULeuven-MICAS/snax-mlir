import argparse
from collections.abc import Sequence

from xdsl.dialects import get_all_dialects
from xdsl.transforms import get_all_passes
from xdsl.xdsl_opt_main import xDSLOptMain

from snaxc.accelerators import AccContext, get_all_accelerators
from snaxc.dialects import get_all_snax_dialects
from snaxc.transforms import get_all_snax_passes
from snaxc.util.snax_memory import L1, L3, TEST


class SNAXOptMain(xDSLOptMain):
    def __init__(
        self,
        description: str = "SNAX modular optimizer driver",
        args: Sequence[str] | None = None,
    ):
        self.available_frontends = {}
        self.available_passes = {}
        self.available_targets = {}

        self.ctx: AccContext = AccContext()

        self.register_all_dialects()
        self.register_all_accelerators()
        self.register_all_frontends()
        self.register_all_passes()
        self.register_all_targets()
        self.register_all_memories()

        # arg handling
        arg_parser = argparse.ArgumentParser(description=description)
        self.register_all_arguments(arg_parser)
        self.args = arg_parser.parse_args(args=args)
        self.ctx.allow_unregistered = self.args.allow_unregistered_dialect
        self.setup_pipeline()

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

    def register_all_accelerators(self):
        for accelerator_name, accelerator_factory in get_all_accelerators().items():
            self.ctx.register_accelerator(accelerator_name, accelerator_factory)

    def register_all_memories(self):
        self.ctx.register_memory(L1)
        self.ctx.register_memory(L3)
        self.ctx.register_memory(TEST)


def main():
    SNAXOptMain().run()


if "__main__" == __name__:
    main()
