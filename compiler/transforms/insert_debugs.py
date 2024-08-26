from dataclasses import dataclass, field

from xdsl.context import MLContext
from xdsl.dialects import builtin, linalg
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint

from compiler.dialects import snax
from compiler.transforms.insert_sync_barrier import InsertSyncBarrierRewriter
from compiler.util.kernel_type import KernelType


@dataclass(frozen=True)
class InsertDebugStatements(RewritePattern):
    """
    Insert debugs :)
    """
    level: str = field(default="L1")

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        kernel_type = KernelType.get_kernel(op)

        if kernel_type == KernelType.QMAC:
            ## matmuls
            debug = snax.Debug(op.inputs[0], op.inputs[1], op.outputs[0], "gemm", "before", self.level)
            debug2 = snax.Debug(op.inputs[0], op.inputs[1], op.outputs[0], "gemm", "after", self.level)
            rewriter.insert_op(debug, InsertPoint.before(op))
            rewriter.insert_op(debug2, InsertPoint.after(op))

        if kernel_type == KernelType.ADD:
            ## bias add
            debug = snax.Debug(op.inputs[0], op.inputs[1], op.outputs[0], "bias", "before", self.level)
            debug2 = snax.Debug(op.inputs[0], op.inputs[1], op.outputs[0], "bias", "after", self.level)
            rewriter.insert_op(debug, InsertPoint.before(op))
            rewriter.insert_op(debug2, InsertPoint.after(op))

        if kernel_type == KernelType.RESCALE:
            ## rescale and clamp
            debug = snax.Debug(op.inputs[0], op.inputs[0], op.outputs[0], "simd", "before", self.level)
            debug2 = snax.Debug(op.inputs[0], op.inputs[0], op.outputs[0], "simd", "after", self.level)
            rewriter.insert_op(debug, InsertPoint.before(op))
            rewriter.insert_op(debug2, InsertPoint.after(op))


class InsertDebugPass(ModulePass):

    name = "insert-debugs"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            InsertDebugStatements(), apply_recursively=False
        ).rewrite_module(op)
