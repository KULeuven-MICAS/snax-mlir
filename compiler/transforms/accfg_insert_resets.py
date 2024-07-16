import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from xdsl.dialects import builtin, scf
from xdsl.ir import Attribute, MLContext, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern

from compiler.dialects import accfg
from compiler.inference.dataflow import (
    get_insertion_points_where_val_dangles,
    uses_through_controlflow,
)

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


@dataclass(frozen=True)
class InsertResetsForDanglingStatesPattern(RewritePattern):
    """
    This inspects all SSA values and inserts a `reset` operation if the value is not consumed by a `reset`,
    `setup` or `scf.yield` operation.

    if `reset_after_await` is given, it inserts a reset operation after the last await of the last launch instead.
    (this might be wonky if there's complex control flow involved, lol)
    """

    reset_after_await: bool

    @ssa_val_rewrite_pattern(accfg.StateType)
    def match_and_rewrite(self, val: SSAValue, rewriter: PatternRewriter, /):
        # get uses through ctrlflow:
        uses = tuple(uses_through_controlflow(val))
        # abort if any use is a reset or another setup op (or returned from control flow)
        # if it's returned from control flow, we should instead worry about the return value.
        if any(
            isinstance(use.operation, accfg.ResetOp | accfg.SetupOp | scf.Yield)
            for use in uses
        ):
            return

        # if reset_after_await is given, reset after the tokens of the launch ops are no longer dangling
        # (i.e. have been awaited)
        if self.reset_after_await:
            vals = []
            for use in uses:
                if isinstance(use.operation, accfg.LaunchOp):
                    vals.append(use.operation.token)
            if not vals:
                vals.append(val)
        else:
            # by default just run on val
            vals = [val]

        for point in get_insertion_points_where_val_dangles(vals):
            rewriter.insert_op(
                accfg.ResetOp(val),
                point,
            )


@dataclass(frozen=True)
class InsertResetsPass(ModulePass):
    """
    Looks for dangling SSA values of type accfg.state and adds an `accfg.reset` operation to reset these states.
    """

    name = "accfg-insert-resets"

    reset_after_await: bool = False

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            InsertResetsForDanglingStatesPattern(self.reset_after_await)
        ).rewrite_module(op)
