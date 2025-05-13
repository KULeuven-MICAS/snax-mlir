from xdsl.context import Context
from xdsl.dialects import builtin, func, scf
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from snaxc.dialects import snax


class InsertMcycleForLoop(RewritePattern):
    """
    Pattern that looks for top-level scf.for ops and inserts a SNAX mcycle op before and after the top-level for loop.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        for opy in op.body.block.ops:
            if isinstance(opy, scf.ForOp):
                rewriter.insert_op(snax.MCycleOp(), InsertPoint.before(opy))
                rewriter.insert_op(snax.MCycleOp(), InsertPoint.after(opy))


class AddMcycleAroundLoopPass(ModulePass):
    """
    Pass to insert an mcycle op before and after all top-level loops inside a function
    """

    name = "test-add-mcycle-around-loop"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            InsertMcycleForLoop(), apply_recursively=False
        ).rewrite_module(op)
