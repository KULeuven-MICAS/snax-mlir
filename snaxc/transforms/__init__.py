from collections.abc import Callable

from xdsl.passes import ModulePass


def get_all_snax_passes() -> dict[str, Callable[[], type[ModulePass]]]:
    """Return the list of all available passes."""

    def get_accfg_config_overlap():
        from snaxc.transforms.accfg_config_overlap import AccfgConfigOverlapPass

        return AccfgConfigOverlapPass

    def get_accfg_dedup():
        from snaxc.transforms.accfg_dedup import AccfgDeduplicate

        return AccfgDeduplicate

    def get_accfg_insert_resets():
        from snaxc.transforms.accfg_insert_resets import InsertResetsPass

        return InsertResetsPass

    def get_accfg_trace_states():
        from snaxc.transforms.convert_linalg_to_accfg import TraceStatesPass

        return TraceStatesPass

    def get_alloc_to_global():
        from snaxc.transforms.alloc_to_global import AllocToGlobalPass

        return AllocToGlobalPass

    def get_clear_memory_space():
        from snaxc.transforms.clear_memory_space import ClearMemorySpace

        return ClearMemorySpace

    def construct_pipeline():
        from snaxc.transforms.pipeline.construct_pipeline import (
            ConstructPipelinePass,
        )

        return ConstructPipelinePass

    def get_convert_accfg_to_csr():
        from snaxc.transforms.convert_accfg_to_csr import ConvertAccfgToCsrPass

        return ConvertAccfgToCsrPass

    def get_convert_dart_to_snax_stream():
        from snaxc.transforms.convert_dart_to_snax_stream import (
            ConvertDartToSnaxStream,
        )

        return ConvertDartToSnaxStream

    def get_convert_kernel_to_linalg():
        from snaxc.transforms.convert_kernel_to_linalg import ConvertKernelToLinalg

        return ConvertKernelToLinalg

    def get_convert_linalg_to_accfg():
        from snaxc.transforms.convert_linalg_to_accfg import ConvertLinalgToAccPass

        return ConvertLinalgToAccPass

    def get_convert_linalg_to_dart():
        from snaxc.transforms.dart.convert_linalg_to_dart import ConvertLinalgToDart

        return ConvertLinalgToDart

    def get_convert_linalg_to_kernel():
        from snaxc.transforms.convert_linalg_to_kernel import ConvertLinalgToKernel

        return ConvertLinalgToKernel

    def get_convert_tosa_to_kernel():
        from snaxc.transforms.convert_tosa_to_kernel import ConvertTosaToKernelPass

        return ConvertTosaToKernelPass

    def get_dart_fuse_operations():
        from snaxc.transforms.dart.dart_fuse_operations import DartFuseOperationsPass

        return DartFuseOperationsPass

    def get_dart_layout_resolution():
        from snaxc.transforms.dart.dart_layout_resolution import (
            DartLayoutResolutionPass,
        )

        return DartLayoutResolutionPass

    def get_dart_scheduler():
        from snaxc.transforms.dart.dart_scheduler import DartSchedulerPass

        return DartSchedulerPass

    def get_dispatch_kernels():
        from snaxc.transforms.dispatch_kernels import DispatchKernels

        return DispatchKernels

    def get_dispatch_regions():
        from snaxc.transforms.dispatch_regions import DispatchRegions

        return DispatchRegions

    def get_insert_accfg_op():
        from snaxc.transforms.insert_accfg_op import InsertAccOp

        return InsertAccOp

    def get_insert_sync_barrier():
        from snaxc.transforms.insert_sync_barrier import InsertSyncBarrier

        return InsertSyncBarrier

    def get_memref_to_snax():
        from snaxc.transforms.memref_to_snax import MemrefToSNAX

        return MemrefToSNAX

    def get_pipeline_canonicalize_for():
        from snaxc.transforms.pipeline.pipeline_canonicalize_for import (
            PipelineCanonicalizeFor,
        )

        return PipelineCanonicalizeFor

    def get_pipeline_duplicate_buffers():
        from snaxc.transforms.pipeline.pipeline_duplicate_buffers import (
            PipelineDuplicateBuffersPass,
        )

        return PipelineDuplicateBuffersPass

    def get_postprocess_mlir():
        from snaxc.transforms.backend.postprocess_mlir import PostprocessPass

        return PostprocessPass

    def get_preprocess_mlir():
        from snaxc.transforms.frontend.preprocess_mlir import PreprocessPass

        return PreprocessPass

    def get_preprocess_mlperf_tiny():
        from snaxc.transforms.frontend.preprocess_mlperf_tiny import (
            PreprocessMLPerfTiny,
        )

        return PreprocessMLPerfTiny

    def get_realize_memref_casts():
        from snaxc.transforms.realize_memref_casts import RealizeMemrefCastsPass

        return RealizeMemrefCastsPass

    def get_reuse_memref_allocs():
        from snaxc.transforms.reuse_memref_allocs import ReuseMemrefAllocs

        return ReuseMemrefAllocs

    def get_set_memory_layout():
        from snaxc.transforms.set_memory_layout import SetMemoryLayout

        return SetMemoryLayout

    def get_set_memory_space():
        from snaxc.transforms.set_memory_space import SetMemorySpace

        return SetMemorySpace

    def get_snax_bufferize():
        from snaxc.transforms.snax_bufferize import SnaxBufferize

        return SnaxBufferize

    def get_snax_copy_to_dma():
        from snaxc.transforms.snax_copy_to_dma import SNAXCopyToDMA

        return SNAXCopyToDMA

    def get_snax_lower_mcycle():
        from snaxc.transforms.snax_lower_mcycle import SNAXLowerMCycle

        return SNAXLowerMCycle

    def get_snax_to_func():
        from snaxc.transforms.snax_to_func import SNAXToFunc

        return SNAXToFunc

    def get_test_add_mcycle_around_loop():
        from snaxc.transforms.test_add_mcycle_around_loop import (
            AddMcycleAroundLoopPass,
        )

        return AddMcycleAroundLoopPass

    def get_test_add_mcycle_around_launch():
        from snaxc.transforms.test.test_add_mcycle_around_launch import (
            AddMcycleAroundLaunch,
        )

        return AddMcycleAroundLaunch

    def get_test_debug_to_func():
        from snaxc.transforms.test.debug_to_func import DebugToFuncPass

        return DebugToFuncPass

    def get_test_insert_debugs():
        from snaxc.transforms.test.insert_debugs import InsertDebugPass

        return InsertDebugPass

    def get_test_remove_memref_copy():
        from snaxc.transforms.test_remove_memref_copy import RemoveMemrefCopyPass

        return RemoveMemrefCopyPass

    def get_unroll_pipeline():
        from snaxc.transforms.pipeline.unroll_pipeline import UnrollPipelinePass

        return UnrollPipelinePass

    return {
        "accfg-config-overlap": get_accfg_config_overlap,
        "accfg-dedup": get_accfg_dedup,
        "accfg-insert-resets": get_accfg_insert_resets,
        "accfg-trace-states": get_accfg_trace_states,
        "alloc-to-global": get_alloc_to_global,
        "clear-memory-space": get_clear_memory_space,
        "construct-pipeline": construct_pipeline,
        "convert-accfg-to-csr": get_convert_accfg_to_csr,
        "convert-dart-to-snax-stream": get_convert_dart_to_snax_stream,
        "convert-kernel-to-linalg": get_convert_kernel_to_linalg,
        "convert-linalg-to-accfg": get_convert_linalg_to_accfg,
        "convert-linalg-to-dart": get_convert_linalg_to_dart,
        "convert-linalg-to-kernel": get_convert_linalg_to_kernel,
        "convert-tosa-to-kernel": get_convert_tosa_to_kernel,
        "dart-fuse-operations": get_dart_fuse_operations,
        "dart-layout-resolution": get_dart_layout_resolution,
        "dart-scheduler": get_dart_scheduler,
        "dispatch-kernels": get_dispatch_kernels,
        "dispatch-regions": get_dispatch_regions,
        "insert-accfg-op": get_insert_accfg_op,
        "insert-sync-barrier": get_insert_sync_barrier,
        "memref-to-snax": get_memref_to_snax,
        "pipeline-canonicalize-for": get_pipeline_canonicalize_for,
        "pipeline-duplicate-buffers": get_pipeline_duplicate_buffers,
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
        "test-debug-to-func": get_test_debug_to_func,
        "test-insert-debugs": get_test_insert_debugs,
        "test-remove-memref-copy": get_test_remove_memref_copy,
        "unroll-pipeline": get_unroll_pipeline,
    }
