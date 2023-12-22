import argparse
from collections.abc import Sequence

from xdsl.ir import MLContext
from xdsl.xdsl_opt_main import xDSLOptMain

from compiler.dialects.snax import Snax
from compiler.dialects.tsl import TSL
from compiler.dialects.tsl import TSL
from compiler.transforms.clear_memory_space import ClearMemorySpace
from compiler.transforms.dispatch_elementwise_mult import DispatchElementWiseMult
from compiler.transforms.dispatch_regions import DispatchRegions
from compiler.transforms.insert_sync_barrier import InsertSyncBarrier
from compiler.transforms.linalg_to_library_call import LinalgToLibraryCall
from compiler.transforms.set_memory_space import SetMemorySpace
from compiler.transforms.snax_copy_to_dma import SNAXCopyToDMA
from compiler.transforms.snax_to_func import SNAXToFunc


class SNAXOptMain(xDSLOptMain):
    def __init__(
        self,
        description: str = "SNAX modular optimizer driver",
        args: Sequence[str] | None = None,
    ):
        self.available_frontends = {}
        self.available_passes = {}
        self.available_targets = {}

        self.ctx = MLContext()
        super().register_all_dialects()
        super().register_all_frontends()
        super().register_all_passes()
        super().register_all_targets()

        ## Add custom dialects & passes
        self.ctx.load_dialect(Snax)
        self.ctx.load_dialect(TSL)
        super().register_pass(DispatchElementWiseMult)
        super().register_pass(LinalgToLibraryCall)
        super().register_pass(SetMemorySpace)
        super().register_pass(InsertSyncBarrier)
        super().register_pass(DispatchRegions)
        super().register_pass(SNAXCopyToDMA)
        super().register_pass(SNAXToFunc)
        super().register_pass(ClearMemorySpace)

        # arg handling
        arg_parser = argparse.ArgumentParser(description=description)
        super().register_all_arguments(arg_parser)
        self.args = arg_parser.parse_args(args=args)

        self.ctx.allow_unregistered = self.args.allow_unregistered_dialect

        super().setup_pipeline()

    pass


def main():
    SNAXOptMain().run()


if "__main__" == __name__:
    main()
