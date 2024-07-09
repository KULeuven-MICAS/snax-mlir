import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from xdsl.dialects import builtin
from xdsl.ir import Attribute, Block, MLContext, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern
from xdsl.rewriter import InsertPoint
from xdsl.traits import IsTerminator

from compiler.dialects import accfg

_RewritePatternT = TypeVar("_RewritePatternT", bound=RewritePattern)


def ssa_val_rewrite_pattern(
    val_type: type[Attribute],
) -> Callable[
    [Callable[[_RewritePatternT, SSAValue, PatternRewriter], None]],
    Callable[[_RewritePatternT, Operation, PatternRewriter], None],
]:
    """
    Expresses a rewrite pattern that acts on SSA Values instead of operations.

    Pass `val_type` and only match on SSA Values of the specified type.
    """

    def wrapper(
        old_match_and_rewrite: Callable[
            [_RewritePatternT, SSAValue, PatternRewriter], None
        ]
    ) -> Callable[[_RewritePatternT, Operation, PatternRewriter], None]:
        # this is the function that actually wraps the match_and_rewrite method

        def match_and_rewrite(
            self: _RewritePatternT, op: Operation, rewriter: PatternRewriter
        ):
            for val in itertools.chain(
                op.results,
                *(block.args for region in op.regions for block in region.blocks),
            ):
                if not isinstance(val.type, val_type):
                    continue
                old_match_and_rewrite(self, val, rewriter)

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

        # grab a the candidate for this block
        candidate = inserts.get(block, None)
        # and the position of the current op in the block
        idx = block.get_operation_index(use.operation)

        # if there is a candidate
        if candidate is not None:
            # and the current op comes before the candidate
            if idx < candidate.idx:
                continue  # skip it
        # if we are a terminator
        if use.operation.has_trait(IsTerminator):
            # don't insert in this block
            inserts.pop(block, None)
            # never insert in this block
            dead_blocks.add(block)
            continue

        # put insertion candidate into candidate dict
        inserts[block] = InsertCandidate(idx, use.operation)

    # return all insertion candidates that we have
    yield from (InsertPoint.after(cd.op) for cd in inserts.values())


class InsertResetsForDanglingStatesPattern(RewritePattern):
    """
    This inspects all SSA values and inserts a `reset` operation if the value has no uses outside of `launch` and
    `reset` operations.
    """

    @ssa_val_rewrite_pattern(accfg.StateType)
    def match_and_rewrite(self, val: SSAValue, rewriter: PatternRewriter, /):
        # abort if
        # any use is a reset or another setup op
        if any(
            isinstance(use.operation, accfg.ResetOp | accfg.SetupOp) for use in val.uses
        ):
            return

        for point in get_insertion_points_where_val_dangles(val):
            rewriter.insert_op(
                accfg.ResetOp(val),
                point,
            )


class InsertResetsPass(ModulePass):
    """
    Looks for dangling SSA values of type accfg.state and adds an `accfg.reset` operation to reset these states.
    """

    name = "accfg-insert-resets"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(InsertResetsForDanglingStatesPattern()).rewrite_module(op)
