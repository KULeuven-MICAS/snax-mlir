from xdsl.context import MLContext
from xdsl.dialects import builtin, transform
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class RemoveSchedule(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: transform.NamedSequenceOp | transform.SequenceOp,
        rewriter: PatternRewriter,
        /,
    ):
        rewriter.erase_matched_op()


class TestTransformDialectEraseSchedule(ModulePass):
    """
    Copy of the test-transform-dialect-erase-schedule pass in MLIR
    """

    name = "test-transform-dialect-erase-schedule"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(RemoveSchedule()).rewrite_module(op)
