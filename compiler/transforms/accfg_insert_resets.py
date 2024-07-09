import itertools
from dataclasses import dataclass

from xdsl.dialects import builtin
from xdsl.ir import Attribute, Block, MLContext, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.traits import IsTerminator

from compiler.dialects import accfg


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


@dataclass(frozen=True, slots=True)
class InsertCandidate:
    idx: int
    op: Operation


def get_insertion_points_where_val_dangles(val: SSAValue):
    """
    Return all insertion points where val dangles after it is used last.

    """
    # if no uses are found, just return the first position where the value is live:
    if not val.uses:
        if isinstance(val.owner, Operation):
            yield InsertPoint.after(val.owner)
        else:
            yield InsertPoint.at_start(val.owner)
        return

    # remember insertion candidates and their positions
    # we need to keep track of the last use of the value in each block it's used in.
    inserts: dict[Block, InsertCandidate] = dict()
    dead_blocks: set[Block] = set()
    # check each use
    for use in val.uses:
        # grab the block
        block = use.operation.parent_block()
        if block is None or block in dead_blocks:
            continue

        candidate = inserts.get(block, None)
        idx = block.get_operation_index(use.operation)

        if candidate is not None:
            if idx < candidate.idx:
                continue
        if use.operation.has_trait(IsTerminator):
            inserts.pop(block, None)
            dead_blocks.add(block)
            continue

        inserts[block] = InsertCandidate(idx, use.operation)

    yield from (InsertPoint.after(cd.op) for cd in inserts.values())


class InsertResetsForDanglingStatesPattern(RewritePattern):
    """
    This inspects all SSA values and inserts a `reset` operation if the value has no uses outside of `launch` and
    `reset` operations.
    """

    @ssa_val_rewrite_pattern(accfg.StateType)
    def match_and_rewrite(self, val: SSAValue, rewriter: PatternRewriter, /):
        if val.uses and all(
            isinstance(use.operation, accfg.SetupOp | accfg.ResetOp) for use in val.uses
        ):
            return

        for point in get_insertion_points_where_val_dangles(val):
            rewriter.insert_op(
                accfg.ResetOp(val),
                point,
            )


class InsertResetsPass(ModulePass):
    """
    Looks for dangling SSA values of type accfg.state

    """

    name = "accfg-insert-resets"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(InsertResetsForDanglingStatesPattern()).rewrite_module(op)
