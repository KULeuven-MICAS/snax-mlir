from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.arith import ConstantOp, DivUIOp, MuliOp, RemUIOp
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.dialects.scf import ForOp, YieldOp
from xdsl.ir import OpResult
from xdsl.irdl import Operand
from xdsl.parser import IntegerAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
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
        new_for.body.block.args[0].replace_by_if(new_iter_var.result, lambda use: use.operation is not new_iter_var)

        # insert the ops
        rewriter.insert_op(new_iter_var, InsertPoint.at_start(new_for.body.block))
        rewriter.replace_op(op, (new_step, new_ub, new_for))


@dataclass
class MergeForLoops(RewritePattern):
    """
    Merges two nested for loops into one.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ForOp, rewriter: PatternRewriter):
        # searching for nested for loops:
        if not isinstance(parent := op.parent_op(), ForOp):
            return

        # lb, ub and step must be index constants
        lb, ub, step = (extract_cst_index(x) for x in (op.lb, op.ub, op.step))
        if lb is None or ub is None or step is None:
            return

        # same for the parent op:
        lb_parent, ub_parent, step_parent = (extract_cst_index(x) for x in (parent.lb, parent.ub, parent.step))
        if lb_parent is None or ub_parent is None or step_parent is None:
            return

        # lb must be 0 and step must be 1:
        if lb != 0 or lb_parent != 0 or step != 1 or step_parent != 1:
            return

        # the new ub of the parent op is ub * ub_parent
        new_parent_ub = ConstantOp.from_int_and_width(ub * ub_parent, IndexType())
        rewriter.insert_op(new_parent_ub, InsertPoint.before(parent))
        new_parent = ForOp(
            parent.lb,
            new_parent_ub,
            parent.step,
            [],
            rewriter.move_region_contents_to_new_regions(parent.body),
        )

        # new parent iter variable is iter // ub
        # create a new op for the div value to make sure it is defined early enough
        div_val = ConstantOp.from_int_and_width(ub, IndexType())
        new_parent_iter = DivUIOp(new_parent.body.block.args[0], div_val)
        new_parent.body.block.args[0].replace_by_if(
            new_parent_iter.result, lambda use: use.operation is not new_parent_iter
        )

        # rewrite parent op
        rewriter.insert_op((div_val, new_parent_iter), InsertPoint.at_start(new_parent.body.block))
        rewriter.replace_op(parent, new_parent)

        # the matched for loop is merged into the parent one, with an iter value of iter // ub
        new_iter = RemUIOp(new_parent.body.block.args[0], div_val)
        op.body.block.args[0].replace_by(new_iter.result)

        rewriter.insert_op(new_iter, InsertPoint.before(op))

        # now the inner for loop can be dismantled, by inlining the block contents
        # remove yield op
        assert isinstance(op.body.block.last_op, YieldOp)
        rewriter.erase_op(op.body.block.last_op)
        rewriter.inline_block(op.body.block, InsertPoint.before(op))
        rewriter.erase_op(op)  # remove remaining empty for loop


@dataclass(frozen=True)
class PipelineCanonicalizeFor(ModulePass):
    name = "pipeline-canonicalize-for"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(GreedyRewritePatternApplier([ChangeForStep(), MergeForLoops()])).rewrite_module(op)
