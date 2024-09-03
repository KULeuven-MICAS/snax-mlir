from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, scf
from xdsl.ir import Block, BlockArgument, Operation, Use
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
        if all(
            between_op in inputs.inputs for between_op in iter_ops_range(launch, op)
        ):
            return

        # and move the setup and inputs to be right behind the launch
        inputs.lazy_move_up(
            parent_block,
            InsertPoint.after(launch),
            rewriter,
        )


class LoopLevelSetupAwaitOverlapPattern(RewritePattern):
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

    1. We grab the first setup op inside the loop, with all dependencies
    2. We insert a copy of the setup op before the loop, replacing dependencies with the loop inputs (%lb)
    3. We insert a copy of the setup af the end of the loop, replacing dependencies with the next iterations variables
       (which we get from the yield / by adding %step to %i)
    4. We erase the original setup op.


    This results in the following IR post-optimization:

    ```
     %s0 = setup ... // loop-initial state
     %l1 = setup from %s0 to ("i" = %lb)
     scf.for (%i = %lb to %ub step %step) iter_args(%l0 = %s0) ... {
       %t = launch(%l0)
       await(%t)
       %i_next = arith.addi %i, %step
       %l1 = setup from %l0 to ("i" = %i_next)
       yield %l1
     }
    ```
    The setup will then be lifted before the await by the BlockLevelSetupAwaitOverlapPattern.

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

        # only apply if there is a launch op in the same block:
        if not any(
            isinstance(use.operation, accfg.LaunchOp) for use in op.out_state.uses
        ):
            return
        if not all(
            launch.operation.parent_block() is op.parent_block()
            for launch in filter(
                lambda x: isinstance(x.operation, accfg.LaunchOp), op.out_state.uses
            )
        ):
            return

        # also grab the yield op, will be needed later
        yield_op = for_op.body.block.last_op
        assert isinstance(yield_op, scf.Yield)

        # if the setup ops input is not the loop-carried state var, don't apply optimization
        if (
            not isinstance(op.in_state.owner, Block)
            or op.in_state.owner.parent_op() is not for_op
        ):
            return
        # grab the index in iter_args, that our state occupies
        assert isinstance(op.in_state, BlockArgument)
        iter_arg_idx = (
            op.in_state.index - 1
        )  # -1 because the first block arg is the loop index

        # 1. We grab the first setup op inside the loop, with all dependencies
        inputs = get_scoped_setup_inputs(
            op,
            for_op.body.block,
        )
        # if we can't resolve dependencies, we are done for! Abort!
        if inputs is None:
            return

        # 2. We insert a copy of the setup op before the loop, replacing dependencies with the loop inputs (%lb)
        setup_before = inputs.copy_with_new_dependent_vals(
            (for_op.lb, *for_op.iter_args)
        )
        setup_before.insert_at_position(
            rewriter,
            InsertPoint.before(for_op),
        )
        # use the result state of the before-loop setup as initial state
        for_op.operands[3 + iter_arg_idx] = setup_before.setup.out_state

        # 3. We insert a copy of the setup af the end of the loop, replacing dependencies with the next iterations
        #    variables (which we get from the yield / by adding %step to %i)
        rewriter.insert_op(
            next_i := arith.Addi(for_op.body.block.args[0], for_op.step),
            InsertPoint.before(yield_op),
        )
        setup_at_end = inputs.copy_with_new_dependent_vals(
            (next_i.result, *yield_op.operands)
        )
        setup_at_end.insert_at_position(
            rewriter,
            InsertPoint.before(yield_op),
        )
        # make sure the yield returns the new state:
        yield_op.operands[iter_arg_idx] = setup_at_end.setup.out_state

        # 4. We erase the original setup op.
        inputs.erase(op.in_state, rewriter)


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
                    LoopLevelSetupAwaitOverlapPattern(),
                ]
            )
        ).rewrite_module(op)


def get_ops_from_uses(uses: set[Use]) -> tuple[Operation, ...]:
    return tuple(set(use.operation for use in uses))
