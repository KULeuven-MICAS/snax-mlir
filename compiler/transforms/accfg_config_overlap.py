from xdsl.dialects import builtin
from xdsl.ir import MLContext, Operation, Use
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.dialects import accfg


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

        # detach the setup op
        op.detach()
        # insert the op right after the launch
        rewriter.insert_op_after(op, launch)


class AccfgConfigOverlapPass(ModulePass):
    """
    This pass moves setup ops upward in code to enable setup-computation overlap.
    """

    name = "accfg-config-overlap"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(BlockLevelSetupAwaitOverlapPattern()).rewrite_module(op)


def get_ops_from_uses(uses: set[Use]) -> tuple[Operation, ...]:
    return tuple(set(use.operation for use in uses))
