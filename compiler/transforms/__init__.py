from collections.abc import Callable

from xdsl.passes import ModulePass


def get_all_snax_passes() -> dict[str, Callable[[], type[ModulePass]]]:
    """Return the list of all available passes."""

    def get_accfg_config_overlap():
        from compiler.transforms.accfg_config_overlap import AccfgConfigOverlapPass

        return AccfgConfigOverlapPass

    def get_accfg_dedup():
        from compiler.transforms.accfg_dedup import AccfgDeduplicate

        return AccfgDeduplicate

    def get_accfg_insert_resets():
        from compiler.transforms.accfg_insert_resets import InsertResetsPass

        return InsertResetsPass

    def get_alloc_to_global():
        from compiler.transforms.alloc_to_global import AllocToGlobalPass

        return AllocToGlobalPass

    def get_clear_memory_space():
        from compiler.transforms.clear_memory_space import ClearMemorySpace

        return ClearMemorySpace

    def get_convert_accfg_to_csr():
        from compiler.transforms.convert_accfg_to_csr import ConvertAccfgToCsrPass

        return ConvertAccfgToCsrPass

    def get_convert_dart_to_snax_stream():
        from compiler.transforms.convert_dart_to_snax_stream import (
            ConvertDartToSnaxStream,
        )

        return ConvertDartToSnaxStream

    def get_convert_kernel_to_linalg():
        from compiler.transforms.convert_kernel_to_linalg import ConvertKernelToLinalg

        return ConvertKernelToLinalg

    def get_convert_linalg_to_accfg():
        from compiler.transforms.convert_linalg_to_accfg import ConvertLinalgToAccPass

        return ConvertLinalgToAccPass

    def get_convert_linalg_to_dart():
        from compiler.transforms.dart.convert_linalg_to_dart import ConvertLinalgToDart

        return ConvertLinalgToDart

    def get_convert_linalg_to_kernel():
        from compiler.transforms.convert_linalg_to_kernel import ConvertLinalgToKernel

        return ConvertLinalgToKernel

    def get_convert_tosa_to_kernel():
        from compiler.transforms.convert_tosa_to_kernel import ConvertTosaToKernelPass

        return ConvertTosaToKernelPass

    def get_dart_fuse_operations():
        from compiler.transforms.dart.dart_fuse_operations import DartFuseOperationsPass

        return DartFuseOperationsPass

    def get_dart_layout_resolution():
        from compiler.transforms.dart.dart_layout_resolution import (
            DartLayoutResolutionPass,
        )

        return DartLayoutResolutionPass

    def get_dart_scheduler():
        from compiler.transforms.dart.dart_scheduler import DartSchedulerPass

        return DartSchedulerPass

    def get_debug_to_func():
        from compiler.transforms.test.debug_to_func import DebugToFuncPass

        return DebugToFuncPass

    def get_dispatch_kernels():
        from compiler.transforms.dispatch_kernels import DispatchKernels

        return DispatchKernels

    def get_dispatch_regions():
        from compiler.transforms.dispatch_regions import DispatchRegions

        return DispatchRegions

    def get_insert_accfg_op():
        from compiler.transforms.insert_accfg_op import InsertAccOp

        return InsertAccOp

    def get_insert_debugs():
        from compiler.transforms.test.insert_debugs import InsertDebugPass

        return InsertDebugPass

    def get_insert_sync_barrier():
        from compiler.transforms.insert_sync_barrier import InsertSyncBarrier

        return InsertSyncBarrier

    def get_memref_to_snax():
        from compiler.transforms.memref_to_snax import MemrefToSNAX

        return MemrefToSNAX

    def get_postprocess_mlir():
        from compiler.transforms.backend.postprocess_mlir import PostprocessPass

        return PostprocessPass

    def get_preprocess_mlir():
        from compiler.transforms.frontend.preprocess_mlir import PreprocessPass

        return PreprocessPass

    def get_preprocess_mlperf_tiny():
        from compiler.transforms.frontend.preprocess_mlperf_tiny import (
            PreprocessMLPerfTiny,
        )

        return PreprocessMLPerfTiny

    def get_realize_memref_casts():
        from compiler.transforms.realize_memref_casts import RealizeMemrefCastsPass

        return RealizeMemrefCastsPass

    def get_reuse_memref_allocs():
        from compiler.transforms.reuse_memref_allocs import ReuseMemrefAllocs

        return ReuseMemrefAllocs

    def get_set_memory_layout():
        from compiler.transforms.set_memory_layout import SetMemoryLayout

        return SetMemoryLayout

    def get_set_memory_space():
        from compiler.transforms.set_memory_space import SetMemorySpace

        return SetMemorySpace

    def get_snax_bufferize():
        from compiler.transforms.snax_bufferize import SnaxBufferize

        return SnaxBufferize

    def get_snax_copy_to_dma():
        from compiler.transforms.snax_copy_to_dma import SNAXCopyToDMA

        return SNAXCopyToDMA

    def get_snax_lower_mcycle():
        from compiler.transforms.snax_lower_mcycle import SNAXLowerMCycle

        return SNAXLowerMCycle

    def get_snax_to_func():
        from compiler.transforms.snax_to_func import SNAXToFunc

        return SNAXToFunc

    def get_test_add_mcycle_around_loop():
        from compiler.transforms.test_add_mcycle_around_loop import (
            AddMcycleAroundLoopPass,
        )

        return AddMcycleAroundLoopPass

    def get_test_add_mcycle_around_launch():
        from compiler.transforms.test.test_add_mcycle_around_launch import (
            AddMcycleAroundLaunch,
        )

        return AddMcycleAroundLaunch

    def get_test_remove_memref_copy():
        from compiler.transforms.test_remove_memref_copy import RemoveMemrefCopyPass

        return RemoveMemrefCopyPass

    def get_trace_states_pass():
        from compiler.transforms.convert_linalg_to_accfg import TraceStatesPass

        return TraceStatesPass

    return {
        "accfg-config-overlap": get_accfg_config_overlap,
        "accfg-dedup": get_accfg_dedup,
        "accfg-insert-resets": get_accfg_insert_resets,
        "alloc-to-global": get_alloc_to_global,
        "clear-memory-space": get_clear_memory_space,
        "convert-accfg-to-csr": get_convert_accfg_to_csr,
        "convert-dart-to-snax_stream": get_convert_dart_to_snax_stream,
        "convert-kernel-to-linalg": get_convert_kernel_to_linalg,
        "convert-linalg-to-accfg": get_convert_linalg_to_accfg,
        "convert-linalg-to-dart": get_convert_linalg_to_dart,
        "convert-linalg-to-kernel": get_convert_linalg_to_kernel,
        "convert-tosa-to-kernel": get_convert_tosa_to_kernel,
        "dart-fuse-operations": get_dart_fuse_operations,
        "dart-layout-resolution": get_dart_layout_resolution,
        "dart-scheduler": get_dart_scheduler,
        "debug-to-func": get_debug_to_func,
        "dispatch-kernels": get_dispatch_kernels,
        "dispatch-regions": get_dispatch_regions,
        "insert-accfg-op": get_insert_accfg_op,
        "insert-debugs": get_insert_debugs,
        "insert-sync-barrier": get_insert_sync_barrier,
        "memref-to-snax": get_memref_to_snax,
        "postprocess": get_postprocess_mlir,
        "preprocess": get_preprocess_mlir,
        "preprocess-mlperftiny": get_preprocess_mlperf_tiny,
        "realize-memref-casts": get_realize_memref_casts,
        "reuse-memref-allocs": get_reuse_memref_allocs,
        "set-memory-layout": get_set_memory_layout,
        "set-memory-space": get_set_memory_space,
        "snax-bufferize": get_snax_bufferize,
        "snax-copy-to-dma": get_snax_copy_to_dma,
        "snax-lower-mcycle": get_snax_lower_mcycle,
        "snax-to-func": get_snax_to_func,
        "test-add-mcycle-around-launch": get_test_add_mcycle_around_launch,
        "test-add-mcycle-around-loop": get_test_add_mcycle_around_loop,
        "test-remove-memref-copy": get_test_remove_memref_copy,
        "trace-states": get_trace_states_pass,
    }
