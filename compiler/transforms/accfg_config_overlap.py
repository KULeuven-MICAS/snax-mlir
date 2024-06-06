from xdsl.dialects import builtin, scf
from xdsl.ir import Block, MLContext, Operation, Use
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


class LooplevelSetupAwaitOverlapPattern(RewritePattern):
    """
    Converts an `scf.for` loop to have setup/launch overlaps

    Given the following IR:
    ```
     %s0 = setup ... // loop-initial state
     scf.for (%i = %lb to %ub step %step) iter_args(%l0 = %s0) ... {
       %l1 = setup from %l0 to ("i" = %i)
       %t = launch(%l1)
       await(%t)
       yield %l1
     }
    ```

    1. We insert a copy of the setup op before the loop, replacing the loop variable with `%lb`.
    2. We then move the setup inside the loop *behind* the launch
    3. The launch now consumes the loop-variable directly

    This results in the following IR post-optimization:

    ```
     %s0 = setup ... // loop-initial state
     %l1 = setup from %l0 to ("i" = %lb)
     scf.for (%i = %lb to %ub step %step) iter_args(%l0 = %s0) ... {
       %t = launch(%l0)
       %l1 = setup from %l0 to ("i" = %i)
       await(%t)
       yield %l1
     }
    ```
    This will result in an additional setup that is unused (in the last iteration of the loop), but seeing as this
    is happening during an await anyway, we won't focus much in it for now.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: accfg.SetupOp, rewriter: PatternRewriter, /):
        # only apply if setup has an input state
        if not op.in_state:
            return
        # only apply if setup is inside a for loop
        for_op = op.parent_op()
        if not isinstance(for_op, scf.For):
            return

        # if the setup ops input is not the loop-carried state var, don't apply optimization
        if (
            not isinstance(op.in_state.owner, Block)
            or op.in_state.owner.parent_op() is not for_op
        ):
            return

        # anton did not implement the rest of this optimisation yet.
        return


class AccfgConfigOverlapPass(ModulePass):
    """
    This pass moves setup ops upward in code to enable setup-computation overlap.
    """

    name = "accfg-config-overlap"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(BlockLevelSetupAwaitOverlapPattern()).rewrite_module(op)


def get_ops_from_uses(uses: set[Use]) -> tuple[Operation, ...]:
    return tuple(set(use.operation for use in uses))
