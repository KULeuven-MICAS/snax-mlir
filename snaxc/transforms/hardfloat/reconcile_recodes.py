from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPattern

from snaxc.dialects import hardfloat


class ReconcileRecodes(RewritePattern):
    """
    Cancel pairs of unrecode->recode ops in the code.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: hardfloat.FnToRecFnOp, rewriter: PatternRewriter, /):
        op_owner = op.operands[0].owner
        if not isinstance(op_owner, hardfloat.RecFnToFnOp):
            return
        for res, inp in zip(op.results, op_owner.operands):
            res.replace_all_uses_with(inp)


class ReconcileRecodesPass(ModulePass):
    """
    Cancel unrecode -> recode patterns
    """

    name = "hardfloat-reconcile-recodes"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    # we need to remove ucc as they interfer with the cancellation pass
                    ReconcileUnrealizedCastsPattern(),
                    ReconcileRecodes(),
                ]
            )
        ).rewrite_module(op)
