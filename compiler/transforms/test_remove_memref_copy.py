from xdsl.context import MLContext
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
        rewriter.erase_matched_op()


class RemoveMemrefCopyPass(ModulePass):
    name = "test-remove-memref-copy"

    """
    Pass that removes all memref copy operations.
    Only to be used for testing purposes.
    """

    def apply(self, ctx: MLContext, module: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            RemoveMemrefCopyPattern(), apply_recursively=False
        ).rewrite_module(module)
