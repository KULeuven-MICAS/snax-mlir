from xdsl.context import MLContext
from xdsl.dialects import builtin, linalg
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.dialects.kernel import Parsable


class LowerLinalgBody(RewritePattern):
    """
    Matches on linalg.generic operations to check if
    their body is kernel op defined in the kernel dialect.
    Replaces the body with the equivalent arith body if this is true.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, linalg_op: linalg.GenericOp, rewriter: PatternRewriter):
        # find the kernel op in linalg body
        if not isinstance(kernel_op := linalg_op.body.block.first_op, Parsable):
            return

        # only works for non-fused kernels (only 1 kernel op)
        if not isinstance(kernel_op.next_op, linalg.YieldOp):
            return

        # replace linalg op
        rewriter.replace_matched_op(
            linalg.GenericOp(
                linalg_op.inputs,
                linalg_op.outputs,
                kernel_op.equivalent_region,
                linalg_op.indexing_maps,
                linalg_op.iterator_types,
                linalg_op.result_types,
                linalg_op.library_call,
                linalg_op.doc,
            )
        )


class ConvertKernelToLinalg(ModulePass):
    name = "convert-kernel-to-linalg"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(LowerLinalgBody()).rewrite_module(op)
