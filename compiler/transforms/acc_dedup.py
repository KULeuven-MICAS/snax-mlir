import sys
from dataclasses import dataclass

from xdsl.dialects import builtin, scf
from xdsl.ir import Block, MLContext, Operation, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.dialects import acc
from compiler.inference.trace_acc_state import infer_state_of


class SimplifyRedundantSetupCalls(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.SetupOp, rewriter: PatternRewriter, /):
        # Step 1: Figure out previous state
        prev_state = infer_state_of(op.in_state) if op.in_state else {}

        # Step 2: Filter setup parameters to remove calls that set the same value again
        new_params: list[tuple[str, SSAValue]] = [
            (name, val) for name, val in op.iter_params() if prev_state.get(name) != val
        ]

        # Step 3: If no new params remain, elide the whole op
        if not new_params:
            # This only happens when:
            #  1) The operation has no input state, and
            #  2) The operation has no parameters it sets
            if op.in_state is None:
                # in this case, we can't elide the operation if it's output state is
                # used otherwise we would break stuff. So we just assume that a setup
                # without parameters returns an "empty" state that assumes nothing.
                return

            op.out_state.replace_by(op.in_state)
            rewriter.erase_matched_op()
            return

        # Step 4: If all parameters change, do nothing
        if len(new_params) == len(op.param_names):
            return

        # Step 5: Replace matched op with reduced version
        rewriter.replace_matched_op(
            acc.SetupOp(
                [val for _, val in new_params],
                [param for param, _ in new_params],
                op.accelerator,
                op.in_state,
            )
        )


class PullSetupOpsOutOfLoops(RewritePattern):
    """
    Tries to pull setup operations out of loop nests by:
    1. Inspecting the source block of all operands
    2. Get all operands that originate not from the block the current operation is in or its descendants (X)
    3. Clone the operation, remove all ops that are not in X
    4. Insert the cloned op right before the loop
    5. Remove all operands from the original op that are in X
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.SetupOp, rewriter: PatternRewriter, /):
        # don't apply to setups not inside for loops
        loop_op = op.parent_op()
        if not isinstance(loop_op, scf.For):
            return

        # we can't handle more than one setup op inside the loop for now:
        if op.out_state not in loop_op.body.block.last_op.operands:
            print(
                "Ignoring loop hoisting for loop with multiple setup ops",
                file=sys.stderr,
            )
            print(f"{loop_op}", file=sys.stderr)
            return

        # maybe this isn't needed
        # TODO: test if this is needed
        if op.in_state is None or op.in_state.owner != loop_op.body.block:
            print(
                "Ignoring setup op because its in_state is not the iter_arg",
                file=sys.stderr,
            )
            return

        # get a list of SSA values that originated in this block (= can't be moved)
        values_originating_from_current_block = tuple(
            val for val in op.values if get_ssa_values_blocks(val) == op.parent_block()
        )
        # list of values that can be moved:
        loop_invariant_values = tuple(
            val for val in op.values if val not in values_originating_from_current_block
        )

        # if all variables are loop-dependent, we have nothing left to do.
        if len(values_originating_from_current_block) == len(op.values):
            return

        # create a new op
        loop_invariant_setups = acc.SetupOp(
            loop_invariant_values,
            (
                op.param_names.data[op.values.index(val)]
                for val in loop_invariant_values
            ),
            op.accelerator,
            in_state=get_initial_value_for_scf_for_lcv(loop_op, op.in_state),
        )
        # insert the new setup op before the loop
        rewriter.insert_op_before(loop_invariant_setups, loop_op)

        # replace loop_invariant_setups.in_state with loop_invariant_setups.out_state in the iter_args of loop_op
        loop_op.operands = tuple(
            (
                val
                if val != loop_invariant_setups.in_state
                else loop_invariant_setups.out_state
            )
            for val in loop_op.operands
        )

        # create a new op to replace the old op
        new_in_loop_setup_op = acc.SetupOp(
            values_originating_from_current_block,
            (
                op.param_names.data[op.values.index(val)]
                for val in values_originating_from_current_block
            ),
            op.accelerator,
            in_state=op.in_state,
        )
        # replace the op with the new "slimmer" operation
        rewriter.replace_matched_op(new_in_loop_setup_op)


class HoistSetupCallsIntoConditionals(RewritePattern):
    """
    Hoists setup calls into preceeding scf.if blocks if possible.

    This will never result in additional setup calls and only
    reduced the number of fields that are set up.
    (proof left as an exercise to the reader)

    Things we need to worry about:

    - we can't insert a setup op before the originally produced state.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.SetupOp, rewriter: PatternRewriter, /):
        # do not apply if our in_state is not an scf.if
        if op.in_state is None or not isinstance(op.in_state.owner, scf.If):
            return
        # grab some helper vars
        old_in_state = op.in_state
        assert isinstance(old_in_state, OpResult)

        # Step 1: Check that it's legal to move:
        # grab all launch op uses of the SSA value produced by the scf.if
        # this will only find things that happen *after* the scf.if, so nothing
        # inside the scf.if regions.
        uses = tuple(
            use for use in op.in_state.uses if isinstance(use.operation, acc.LaunchOp)
        )
        # if we have some launch ops, we need to investigate further:
        for use in uses:
            # grab the launch operation
            launch_op = use.operation
            # check that it is in the same block (if not bail out)
            # if it is not in the same block, it means that the launch is nested:
            #   scf.if -> some_op { launch } -> setup
            # properly inferring where the launch op is, is out of scope for now.
            # we always bail out in this case so that we don't break things.
            # TODO: better check this case?
            if launch_op.parent_block() is not op.parent_block():
                return
            # if we share a block, figure out positions and compare:
            block = launch_op.parent_block()
            assert block is not None
            # launch op index < our index => we have a launch between us and the if
            # that means we can't hoist.
            if block.get_operation_index(launch_op) < block.get_operation_index(op):
                return

        # Step 2: Clone the op into the end of both branches
        for region in op.in_state.owner.regions:
            # grab the yield op:
            yield_op = region.block.last_op
            assert isinstance(yield_op, scf.Yield)
            new_in_state = yield_op.operands[old_in_state.index]

            # insert a copy of the setup op but replace the
            # original in_state with the yielded state of the region
            rewriter.insert_op_before(
                new_setup := op.clone(value_mapper={op.in_state: new_in_state}),
                yield_op,
            )
            # replace the yield op with a new yield op that yields the
            # newly produced state
            rewriter.replace_op(
                yield_op,
                yield_op.clone(value_mapper={new_in_state: new_setup.out_state}),
            )

        # erase the op, replacing all uses of its out state by
        # its in_state.
        op.out_state.replace_by(op.in_state)
        rewriter.erase_matched_op()


@dataclass(frozen=True)
class AccDeduplicate(ModulePass):
    """
    Reduce the number of parameters in setup calls by inferring previously
    set up values and carefully moving setup calls around.
    """

    name = "acc-dedup"

    hoist: bool = True

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        patterns = [
            SimplifyRedundantSetupCalls(),
            PullSetupOpsOutOfLoops(),
        ]

        if self.hoist:
            patterns.append(HoistSetupCallsIntoConditionals())

        PatternRewriteWalker(
            GreedyRewritePatternApplier(patterns),
            walk_reverse=True,
        ).rewrite_module(op)


def get_ssa_values_blocks(val: SSAValue) -> Block:
    """
    Returns the block in which an SSA vlaue was created.
    """
    match val.owner:
        case Block() as block:
            return block
        case Operation() as op:
            return op.parent_block()
        case unknown:
            raise ValueError(f"Unknown value owner: {unknown}")


def get_initial_value_for_scf_for_lcv(loop: scf.For, var: SSAValue) -> SSAValue:
    """
    Given a loop-carried variable inside an scf for loop as the block argument,
    return the SSA value that is passed as the initial value to it.
    """
    if var not in loop.body.block.args:
        raise ValueError(
            f"Given value {var} not a block argument of the for loop {loop}!"
        )
    idx = loop.body.block.args.index(var) - 1
    return loop.iter_args[idx]
