from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from snaxc.dialects import accfg, snax


class InsertBeforeLaunch(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: accfg.LaunchOp, rewriter: PatternRewriter, /):
        rewriter.insert_op(snax.MCycleOp(), InsertPoint.before(op))


class InsertAfterAwait(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: accfg.AwaitOp, rewriter: PatternRewriter, /):
        rewriter.insert_op(snax.MCycleOp(), InsertPoint.after(op))


class AddMcycleAroundLaunch(ModulePass):
    """
    Pass to insert an mcycle op before all accfg.launches and after all accfg.awaits
    """

    name = "test-add-mcycle-around-launch"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            InsertBeforeLaunch(), apply_recursively=False
        ).rewrite_module(op)
        PatternRewriteWalker(
            InsertAfterAwait(), apply_recursively=False
        ).rewrite_module(op)
