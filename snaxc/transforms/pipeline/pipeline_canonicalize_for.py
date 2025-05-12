from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.arith import ConstantOp, MuliOp
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.dialects.scf import ForOp
from xdsl.ir import OpResult
from xdsl.irdl import Operand
from xdsl.parser import IntegerAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa


def extract_cst_index(operand: Operand) -> int | None:
    if not isinstance(operand, OpResult):
        return None
    if not isinstance(cst_op := operand.op, ConstantOp):
        return None
    if not isa(cst_value := cst_op.value, IntegerAttr[IndexType]):
        return None
    return cst_value.value.data


@dataclass
class ChangeForStep(RewritePattern):
    """
    Rewrites a for loop to use step 1.
    Currently supported are for loops:
        - without index_args
        - step, ub and lb defined as constants and
        - lb == 0
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ForOp, rewriter: PatternRewriter) -> None:
        # iter args is not supported
        if len(op.iter_args) != 0:
            return

        # lb, ub and step must be index constants
        lb, ub, step = (extract_cst_index(x) for x in (op.lb, op.ub, op.step))
        if lb is None or ub is None or step is None:
            return

        # lb must be 0
        if lb != 0:
            return

        # step must not already be 1
        if step == 1:
            return

        # otherwise, replace op with a new one that uses step 1 and ub = ub // step
        new_step = ConstantOp.from_int_and_width(1, IndexType())
        new_ub = ConstantOp.from_int_and_width(ub // step, IndexType())
        new_for = ForOp(
            op.lb,
            new_ub,
            new_step,
            [],
            rewriter.move_region_contents_to_new_regions(op.body),
        )

        # compute new iteration variable
        new_iter_var = MuliOp(op.step, new_for.body.block.args[0])
        new_for.body.block.args[0].replace_by_if(
            new_iter_var.result, lambda use: use.operation is not new_iter_var
        )

        # insert the ops
        rewriter.insert_op(new_iter_var, InsertPoint.at_start(new_for.body.block))
        rewriter.replace_matched_op((new_step, new_ub, new_for))


@dataclass(frozen=True)
class PipelineCanonicalizeFor(ModulePass):
    name = "pipeline-canonicalize-for"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(ChangeForStep()).rewrite_module(op)
