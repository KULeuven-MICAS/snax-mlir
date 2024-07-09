import itertools

from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint

from compiler.dialects import accfg
from xdsl.ir import Operation, MLContext, Attribute, SSAValue, Block
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
)


def ssa_val_rewrite_pattern(val_type: type[Attribute]):
    def wrapper(fun):
        seen: set[SSAValue] = set()

        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            for val in itertools.chain(
                op.results,
                *(block.args for region in op.regions for block in region.blocks),
            ):
                if not isinstance(val.type, val_type):
                    continue
                if val in seen:
                    continue
                seen.add(val)
                fun(self, val, rewriter)

        return match_and_rewrite

    return wrapper


class InsertResetsForDanglingStatesPattern(RewritePattern):
    @ssa_val_rewrite_pattern(accfg.StateType)
    def match_and_rewrite(self, val: SSAValue, rewriter: PatternRewriter, /):
        if val.uses:
            return
        if isinstance(val.owner, Operation):
            rewriter.insert_op(accfg.ResetOp(val), InsertPoint.after(val.owner))
        else:
            rewriter.insert_op(accfg.ResetOp(val), InsertPoint.at_start(val.owner))


class InsertResetsPass(ModulePass):
    """
    Looks for dangling SSA values of type accfg.state

    """

    name = "accfg-insert-resets"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(InsertResetsForDanglingStatesPattern()).rewrite_module(op)
