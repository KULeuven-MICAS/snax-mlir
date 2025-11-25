from xdsl.context import Context
from xdsl.dialects import builtin, memref
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class RemoveMemrefCopyPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, memref_copy: memref.CopyOp, rewriter: PatternRewriter):
        rewriter.erase_op(memref_copy)


class RemoveMemrefCopyPass(ModulePass):
    name = "test-remove-memref-copy"

    """
    Pass that removes all memref copy operations.
    Only to be used for testing purposes.
    """

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(RemoveMemrefCopyPattern(), apply_recursively=False).rewrite_module(op)
