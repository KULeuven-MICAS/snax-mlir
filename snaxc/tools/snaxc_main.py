import argparse
import sys
from collections.abc import Sequence

import yaml
from xdsl.dialects import get_all_dialects
from xdsl.dialects.builtin import ModuleOp
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PassPipeline
from xdsl.printer import Printer
from xdsl.tools.command_line_tool import CommandLineTool
from xdsl.transforms.canonicalize import CanonicalizePass

from snaxc.accelerators.acc_context import AccContext
from snaxc.dialects import get_all_snax_dialects
from snaxc.tools.config_parser import parse_config
from snaxc.transforms.alloc_to_global import AllocToGlobalPass
from snaxc.transforms.backend.postprocess_mlir import PostprocessPass
from snaxc.transforms.clear_memory_space import ClearMemorySpace
from snaxc.transforms.convert_accfg_to_csr import ConvertAccfgToCsrPass
from snaxc.transforms.convert_dart_to_snax_stream import ConvertDartToSnaxStream
from snaxc.transforms.convert_linalg_to_accfg import ConvertLinalgToAccPass
from snaxc.transforms.convert_linalg_to_kernel import ConvertLinalgToKernel
from snaxc.transforms.convert_memref_to_arith import ConvertMemrefToArithPass
from snaxc.transforms.dart.convert_linalg_to_dart import ConvertLinalgToDart
from snaxc.transforms.dart.dart_fuse_operations import DartFuseOperationsPass
from snaxc.transforms.dart.dart_layout_resolution import DartLayoutResolutionPass
from snaxc.transforms.dart.dart_scheduler import DartSchedulerPass
from snaxc.transforms.dispatch_kernels import DispatchKernels
from snaxc.transforms.dispatch_regions import DispatchRegions
from snaxc.transforms.frontend.frontend_transform import FrontendTransformPass
from snaxc.transforms.frontend.preprocess_mlir import PreprocessPass
from snaxc.transforms.fuse_accumulation_memrefs import FuseAccumulationMemrefsPass
from snaxc.transforms.insert_accfg_op import InsertAccOp
from snaxc.transforms.insert_sync_barrier import InsertSyncBarrier
from snaxc.transforms.memref_to_snax import MemrefToSNAX
from snaxc.transforms.pipeline.construct_pipeline import ConstructPipelinePass
from snaxc.transforms.pipeline.pipeline_canonicalize_for import PipelineCanonicalizeFor
from snaxc.transforms.pipeline.pipeline_duplicate_buffers import PipelineDuplicateBuffersPass
from snaxc.transforms.pipeline.unroll_pipeline import UnrollPipelinePass
from snaxc.transforms.realize_memref_casts import RealizeMemrefCastsPass
from snaxc.transforms.reuse_memref_allocs import ReuseMemrefAllocs
from snaxc.transforms.set_memory_layout import SetMemoryLayout
from snaxc.transforms.set_memory_space import SetMemorySpace
from snaxc.transforms.snax_allocate import SnaxAllocatePass
from snaxc.transforms.snax_bufferize import SnaxBufferize
from snaxc.transforms.snax_copy_to_dma import SNAXCopyToDMA
from snaxc.transforms.snax_lower_mcycle import SNAXLowerMCycle
from snaxc.transforms.snax_to_func import SNAXToFunc
from snaxc.transforms.test.debug_to_func import DebugToFuncPass
from snaxc.transforms.test.insert_debugs import InsertDebugPass
from snaxc.transforms.test.test_add_mcycle_around_launch import AddMcycleAroundLaunch


class SNAXCMain(CommandLineTool):
    def __init__(
        self,
        description: str = "SNAX-MLIR Compiler",
        args: Sequence[str] | None = None,
    ):
        # arg handling
        arg_parser = argparse.ArgumentParser(description=description)
        self.register_all_arguments(arg_parser)
        self.args = arg_parser.parse_args(args=args)

        self.load_config()
        self.register_all_dialects()
        self.setup_pipeline()

    def register_all_dialects(self):
        all_dialects = get_all_dialects()
        # FIXME: override upstream accfg and stream dialect.
        all_dialects.pop("accfg", None)
        all_dialects.pop("stream", None)
        all_dialects.update(get_all_snax_dialects())
        for dialect_name, dialect_factory in all_dialects.items():
            self.ctx.register_dialect(dialect_name, dialect_factory)

    def load_config(self):
        # read config file
        if self.args.config is not None:
            with open(self.args.config) as f:
                config = yaml.safe_load(f)
            context = parse_config(config)
            context.allow_unregistered = True
            self.ctx = context
        else:
            self.ctx = AccContext(allow_unregistered=True)

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

        arg_parser.add_argument("input_file", type=str, nargs="?", help="path to input file")

        arg_parser.add_argument("-o", "--output-file", type=str, required=True, help="path to output file")

        arg_parser.add_argument(
            "-c",
            "--config",
            type=str,
            help="path to the accelerator configuration file",
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

        arg_parser.add_argument(
            "--alloc-mode",
            choices=["static", "dynamic", "minimalloc", "auto"],
            default="auto",
            help="Select memory allocation scheme",
        )

        arg_parser.add_argument(
            "--debug",
            default=False,
            action="store_true",
            help="insert debugging function calls around accelerator operations",
        )

        arg_parser.add_argument(
            "--no-frontend",
            default=False,
            action="store_true",
            help="disable all frontend passes",
        )

        arg_parser.add_argument(
            "--no-backend",
            default=False,
            action="store_true",
            help="disable all backend passes",
        )

        arg_parser.add_argument(
            "--add-mcycle",
            default=False,
            action="store_true",
            help="add mcycle around accfg.launch ops",
        )

    def setup_pipeline(self):
        """
        Creates a pipeline that consists of all the passes specified.

        Fails, if not all passes are registered.
        """

        assert isinstance(self.ctx, AccContext), "Context must be an AccContext instance"

        def callback(previous_pass: ModulePass, module: ModuleOp, next_pass: ModulePass) -> None:
            module.verify()
            if self.args.print_between_passes:
                print(f"IR after {previous_pass.name}:")
                printer = Printer(stream=sys.stdout)
                printer.print_op(module)
                print("\n\n\n")

        pass_pipeline: list[ModulePass] = []

        # Transform passes:
        pass_pipeline.append(FrontendTransformPass())

        # Frontend passes:
        if not self.args.no_frontend:
            pass_pipeline.append(PreprocessPass())

        # Insert accfg operations based on accelerators registered in the AccContext:
        for accelerator in self.ctx.registered_accelerator_names:
            pass_pipeline.append(InsertAccOp(accelerator))

        # Standard lowering pipeline:
        pass_pipeline.append(ConvertLinalgToKernel())
        pass_pipeline.append(DispatchKernels())
        pass_pipeline.append(ConvertLinalgToDart())
        pass_pipeline.append(DartFuseOperationsPass())
        if not self.args.no_frontend:
            pass_pipeline.append(SnaxBufferize())
        if self.args.debug:
            pass_pipeline.append(InsertDebugPass())
        pass_pipeline.append(FuseAccumulationMemrefsPass())
        pass_pipeline.append(AllocToGlobalPass())
        pass_pipeline.append(SetMemorySpace())
        pass_pipeline.append(DartSchedulerPass())
        pass_pipeline.append(SetMemoryLayout())
        if self.args.debug:
            pass_pipeline.append(InsertDebugPass())
        pass_pipeline.append(RealizeMemrefCastsPass())
        pass_pipeline.append(ReuseMemrefAllocs())
        pass_pipeline.append(InsertSyncBarrier())
        pass_pipeline.append(PipelineCanonicalizeFor())
        pass_pipeline.append(ConstructPipelinePass())
        pass_pipeline.append(PipelineDuplicateBuffersPass())
        pass_pipeline.append(UnrollPipelinePass())
        pass_pipeline.append(MemrefToSNAX())
        pass_pipeline.append(CanonicalizePass())
        pass_pipeline.append(SnaxAllocatePass(self.args.alloc_mode))
        pass_pipeline.append(InsertSyncBarrier())
        pass_pipeline.append(DispatchRegions())
        pass_pipeline.append(DartLayoutResolutionPass())
        pass_pipeline.append(ConvertDartToSnaxStream())
        pass_pipeline.append(ConvertLinalgToAccPass())
        if self.args.add_mcycle:
            pass_pipeline.append(AddMcycleAroundLaunch())
        pass_pipeline.append(ConvertAccfgToCsrPass())
        pass_pipeline.append(SNAXCopyToDMA())
        pass_pipeline.append(SNAXToFunc())
        pass_pipeline.append(ConvertMemrefToArithPass())
        pass_pipeline.append(SNAXLowerMCycle())
        if self.args.debug:
            pass_pipeline.append(DebugToFuncPass())
        pass_pipeline.append(ClearMemorySpace())
        pass_pipeline.append(CanonicalizePass())

        # Convert to llvm:
        if not self.args.no_backend:
            pass_pipeline.append(PostprocessPass())

        # Initialize pipeline
        self.pipeline = PassPipeline(tuple(pass_pipeline), callback)


def main():
    SNAXCMain().run()


if "__main__" == __name__:
    main()
