from xdsl.dialects import builtin
from xdsl.ir import MLContext, Operation, Use
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from compiler.dialects import accfg
from compiler.inference.helpers import iter_ops_range
from compiler.inference.scoped_setups import get_scoped_setup_inputs


class BlockLevelSetupAwaitOverlapPattern(RewritePattern):
    """
    Inspects a single block for the following structure:

    ```
     %token = accfg.launch %state ...
     // ...
     %next_state = accfg.setup ... from %state ...
    ```

    And moves the setup op right behind the launch, iff:
    - These two ops are the only two uses of the `%state` SSA value
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: accfg.SetupOp, rewriter: PatternRewriter, /):
        # ignore setup without an input state
        if not op.in_state:
            return
        # only apply if there are exactly two uses of the state
        ops_uses = get_ops_from_uses(op.in_state.uses)
        if len(ops_uses) != 2:
            return
        # check that one use is this op, and the other is a launch op
        # and then grab a reference to the launch op
        op1, op2 = ops_uses
        launch: accfg.LaunchOp | None = None
        if op1 == op and isinstance(op2, accfg.LaunchOp):
            launch = op2
        elif op2 == op and isinstance(op1, accfg.LaunchOp):
            launch = op1
        else:
            return

        # check that the launch and the setup are on the same block level
        if launch.parent_block() != op.parent_block():
            return

        # abort if the launch op has already been moved (prevent forever loops)
        if launch.next_op is op:
            return

        # grab the parent block, which can't be none (we know that)
        parent_block = op.parent_block()
        assert parent_block is not None

        # grab the setup op with all inputs in this block
        inputs = get_scoped_setup_inputs(op, parent_block)

        # if we have immovable inputs, abort
        # TODO: it is fine if the immovable inputs are *before* the launch op anyway, as we only
        #       care about ops up until the launch op. We can implement this later if it's a problem.
        if inputs is None:
            return

        # if all the ops between launch and setup are input ops, then there's nothing to move!
        if all(between_op in inputs.inputs for between_op in iter_ops_range(launch, op)):
            return

        inputs.lazy_move_up(  # and move the setup and inputs to be right behind the launch
            parent_block,
            InsertPoint.after(launch),
            rewriter,
        )


class AccfgConfigOverlapPass(ModulePass):
    """
    This pass moves setup ops upward in code to enable setup-computation overlap.
    """

    name = "accfg-config-overlap"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    BlockLevelSetupAwaitOverlapPattern(),
                ]
            )
        ).rewrite_module(op)


def get_ops_from_uses(uses: set[Use]) -> tuple[Operation, ...]:
    return tuple(set(use.operation for use in uses))
