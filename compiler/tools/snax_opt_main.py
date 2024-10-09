import argparse
from collections.abc import Sequence

from xdsl.context import MLContext
from xdsl.xdsl_opt_main import xDSLOptMain

from compiler.dialects.accfg import ACCFG
from compiler.dialects.kernel import Kernel
from compiler.dialects.snax import Snax
from compiler.dialects.snax_stream import SnaxStream
from compiler.dialects.stream import Stream
from compiler.dialects.test.debug import Debug
from compiler.dialects.tsl import TSL
from compiler.transforms.accfg_config_overlap import AccfgConfigOverlapPass
from compiler.transforms.accfg_dedup import AccfgDeduplicate
from compiler.transforms.accfg_insert_resets import InsertResetsPass
from compiler.transforms.clear_memory_space import ClearMemorySpace
from compiler.transforms.convert_accfg_to_csr import ConvertAccfgToCsrPass
from compiler.transforms.convert_kernel_to_linalg import ConvertKernelToLinalg
from compiler.transforms.convert_linalg_to_accfg import (
    ConvertLinalgToAccPass,
    TraceStatesPass,
)
from compiler.transforms.convert_linalg_to_kernel import ConvertLinalgToKernel
from compiler.transforms.convert_linalg_to_stream import ConvertLinalgToStream
from compiler.transforms.convert_tosa_to_kernel import ConvertTosaToKernelPass
from compiler.transforms.dispatch_kernels import DispatchKernels
from compiler.transforms.dispatch_regions import DispatchRegions
from compiler.transforms.frontend.preprocess_mlperf_tiny import PreprocessMLPerfTiny
from compiler.transforms.insert_accfg_op import InsertAccOp
from compiler.transforms.insert_sync_barrier import InsertSyncBarrier
from compiler.transforms.linalg_to_library_call import LinalgToLibraryCall
from compiler.transforms.memref_to_snax import MemrefToSNAX
from compiler.transforms.realize_memref_casts import RealizeMemrefCastsPass
from compiler.transforms.reuse_memref_allocs import ReuseMemrefAllocs
from compiler.transforms.set_memory_layout import SetMemoryLayout
from compiler.transforms.set_memory_space import SetMemorySpace
from compiler.transforms.snax_copy_to_dma import SNAXCopyToDMA
from compiler.transforms.snax_lower_mcycle import SNAXLowerMCycle
from compiler.transforms.snax_to_func import SNAXToFunc
from compiler.transforms.stream_snaxify import StreamSnaxify
from compiler.transforms.test.debug_to_func import DebugToFuncPass
from compiler.transforms.test.insert_debugs import InsertDebugPass
from compiler.transforms.test.test_add_mcycle_around_launch import AddMcycleAroundLaunch
from compiler.transforms.test_add_mcycle_around_loop import AddMcycleAroundLoopPass
from compiler.transforms.test_remove_memref_copy import RemoveMemrefCopyPass


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
        # FIXME: override upstream accfg dialect. Remove this after upstreaming full downstream accfg dialect.
        self.ctx._registered_dialects.pop("accfg", None)  # pyright: ignore
        # Warning: overrides upstream stream dialect.
        self.ctx._registered_dialects.pop("stream", None)  # pyright: ignore

        self.ctx.load_dialect(Snax)
        self.ctx.load_dialect(TSL)
        self.ctx.load_dialect(Kernel)
        self.ctx.load_dialect(ACCFG)
        self.ctx.load_dialect(SnaxStream)
        self.ctx.load_dialect(Debug)
        self.ctx.load_dialect(Stream)
        super().register_pass(DispatchKernels.name, lambda: DispatchKernels)
        super().register_pass(LinalgToLibraryCall.name, lambda: LinalgToLibraryCall)
        super().register_pass(SetMemorySpace.name, lambda: SetMemorySpace)
        super().register_pass(SetMemoryLayout.name, lambda: SetMemoryLayout)
        super().register_pass(InsertAccOp.name, lambda: InsertAccOp)
        super().register_pass(InsertSyncBarrier.name, lambda: InsertSyncBarrier)
        super().register_pass(DispatchRegions.name, lambda: DispatchRegions)
        super().register_pass(SNAXCopyToDMA.name, lambda: SNAXCopyToDMA)
        super().register_pass(SNAXToFunc.name, lambda: SNAXToFunc)
        super().register_pass(SNAXLowerMCycle.name, lambda: SNAXLowerMCycle)
        super().register_pass(ClearMemorySpace.name, lambda: ClearMemorySpace)
        super().register_pass(
            RealizeMemrefCastsPass.name, lambda: RealizeMemrefCastsPass
        )
        super().register_pass(InsertResetsPass.name, lambda: InsertResetsPass)
        super().register_pass(MemrefToSNAX.name, lambda: MemrefToSNAX)
        super().register_pass(AccfgDeduplicate.name, lambda: AccfgDeduplicate)
        super().register_pass(
            ConvertLinalgToAccPass.name, lambda: ConvertLinalgToAccPass
        )
        super().register_pass(TraceStatesPass.name, lambda: TraceStatesPass)
        super().register_pass(ConvertAccfgToCsrPass.name, lambda: ConvertAccfgToCsrPass)
        super().register_pass(
            AccfgConfigOverlapPass.name, lambda: AccfgConfigOverlapPass
        )
        super().register_pass(StreamSnaxify.name, lambda: StreamSnaxify)
        super().register_pass(ReuseMemrefAllocs.name, lambda: ReuseMemrefAllocs)
        super().register_pass(RemoveMemrefCopyPass.name, lambda: RemoveMemrefCopyPass)
        super().register_pass(
            AddMcycleAroundLoopPass.name, lambda: AddMcycleAroundLoopPass
        )
        super().register_pass(ConvertLinalgToKernel.name, lambda: ConvertLinalgToKernel)
        super().register_pass(ConvertKernelToLinalg.name, lambda: ConvertKernelToLinalg)
        super().register_pass(
            ConvertTosaToKernelPass.name, lambda: ConvertTosaToKernelPass
        )
        super().register_pass(InsertDebugPass.name, lambda: InsertDebugPass)
        super().register_pass(DebugToFuncPass.name, lambda: DebugToFuncPass)
        super().register_pass(PreprocessMLPerfTiny.name, lambda: PreprocessMLPerfTiny)
        super().register_pass(AddMcycleAroundLaunch.name, lambda: AddMcycleAroundLaunch)
        super().register_pass(ConvertLinalgToStream.name, lambda: ConvertLinalgToStream)

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
