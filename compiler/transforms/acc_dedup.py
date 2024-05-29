from dataclasses import dataclass

from xdsl.dialects import builtin, scf
from xdsl.ir import (
    Block,
    BlockArgument,
    MLContext,
    Operation,
    OpResult,
    SSAValue,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import Pure

from compiler.dialects import acc
from compiler.inference.trace_acc_state import (
    all_setup_ops_in_region,
    infer_state_of,
)


class SimplifyRedundantSetupCalls(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.SetupOp, rewriter: PatternRewriter, /):
        # Step 1: Figure out previous state
        prev_state = infer_state_of(op.in_state) if op.in_state else {}

        # Step 2: Filter setup parameters to remove calls that set the same value again
        new_params: list[tuple[str, SSAValue]] = [
            (name, val) for name, val in op.iter_params() if prev_state.get(name) != val
        ]

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


class MergeSetupOps(RewritePattern):
    """
    Find two setup ops with only pure ops in between and merge them.

    Merge them into one setup op.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.SetupOp, rewriter: PatternRewriter, /):
        prev_op = op.prev_op
        while prev_op is not None:
            # if we encounter a setup op for the same accelerator, we continue
            if (
                isinstance(prev_op, acc.SetupOp)
                and prev_op.accelerator == op.accelerator
            ):
                break
            # if we encounter a not-pure op, we abort
            if not prev_op.has_trait(Pure):
                return
            prev_op = prev_op.prev_op
        if prev_op is None:
            return

        state = dict(prev_op.iter_params())
        state.update(dict(op.iter_params()))

        rewriter.replace_op(
            prev_op,
            new_setup := acc.SetupOp(
                state.values(), state.keys(), op.accelerator, prev_op.in_state
            ),
        )
        op.out_state.replace_by(new_setup.out_state)

        rewriter.erase_matched_op()


class ElideEmptySetupOps(RewritePattern):
    """
    remove setup ops that set zero parameters, but only if they have both a in- and an out-state
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.SetupOp, rewriter: PatternRewriter, /):
        if len(op.values) == 0 and op.in_state is not None and op.out_state is not None:
            op.out_state.replace_by(op.in_state)
            rewriter.erase_matched_op()


class PullSetupOpsOutOfLoops(RewritePattern):
    """
    Tries to pull setup operations out of loop nests by:
    - Creating a list of values that are safe to hoist:
        - safe to hoist means, that the field is set to the same value every time
        - And that that value originates from outside the loop
    - Inserting a new setup op directly in front of the loop, containing only the safe fields
    - Using that setups state as the new input-state for the loop.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.SetupOp, rewriter: PatternRewriter, /):
        # don't apply to setups not inside for loops
        loop_op = op.parent_op()
        if not isinstance(loop_op, scf.For):
            return

        # only do this for the first setup op in the loop
        if op.in_state is None or op.in_state.owner != loop_op.body.block:
            return

        # iterate over all setups inside this loop and check if their values are loop-invariant or not
        # loop invariant values
        safe_values: set[str] = set()
        # loop dependent values, or value set to multiple different values
        unsafe_vals: set[str] = set()
        # remember the values each field of the accelerator is set to, so that we can determine if they are the same
        acc_fields_to_values: dict[str, SSAValue] = {}
        # iterate over all the setup ops in the region and inspect their values
        for setup in all_setup_ops_in_region(loop_op.body, op.accelerator.data):
            for key, val in setup.items():
                # a value is "unsafe" if it originates inside the loop even on one setup call
                if val_is_defined_in_block(val, loop_op.body.block):
                    unsafe_vals.add(key)
                # and also if it changes at any point in the loop
                elif key in acc_fields_to_values and acc_fields_to_values[key] != val:
                    unsafe_vals.add(key)
                else:
                    # we assume this to be a safe value, but we may encounter a new setup op
                    # that suddenly makes this unsafe. This is handled later when we subtract
                    # all unsafe vals from the safe vals.
                    acc_fields_to_values[key] = val
                    safe_values.add(key)
        # remove all unsafe vals form potentially safe values
        # also pick a deterministic, fixed order for the rest of the rewrite
        loop_invariant_options = tuple(sorted(safe_values - unsafe_vals))

        # nothing to do if everything is loop dependent
        if not loop_invariant_options:
            return

        # create a new op
        loop_invariant_setups = acc.SetupOp(
            (acc_fields_to_values[key] for key in loop_invariant_options),
            loop_invariant_options,
            op.accelerator,
            in_state=get_initial_value_for_scf_for_lcv(loop_op, op.in_state),
        )
        # insert the new setup op before the loop
        rewriter.insert_op_before(loop_invariant_setups, loop_op)

        # replace loop_invariant_setups.in_state with loop_invariant_setups.out_state in the iter_args of loop_op
        # loop_invariant_setups.in_state is the previous input state to the for loop (as determined by a call
        # to get_initial_value_for_scf_for_lcv)
        loop_op.operands = tuple(
            (
                val
                if val != loop_invariant_setups.in_state
                else loop_invariant_setups.out_state
            )
            for val in loop_op.operands
        )


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
            MergeSetupOps(),
            ElideEmptySetupOps(),
        ]

        if self.hoist:
            patterns.append(HoistSetupCallsIntoConditionals())

        PatternRewriteWalker(
            GreedyRewritePatternApplier(patterns),
            walk_reverse=True,
        ).rewrite_module(op)


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


def val_is_defined_in_block(val: SSAValue, block: Block) -> bool:
    """
    Check if val is defined in block or in blocks nested in block.
    """
    if isinstance(val, BlockArgument):
        block_ptr: Block | None = val.owner
        # walk upwards until we hit block or None
        while block_ptr is not None and block_ptr != block:
            # walk upwards
            block_ptr = block_ptr.parent_block()
        # we either ran out of blocks
        return block_ptr is not None
    elif isinstance(val, OpResult):
        op_ptr: Operation | None = val.owner
        # walk up until we either hit the right block, or run out of parents
        while op_ptr is not None and op_ptr.parent_block() != block:
            op_ptr = op_ptr.parent_op()
        # iff we didn't hit the right block, ptr must be none
        return op_ptr is not None
    else:
        raise ValueError(f"Unsupported SSA Value type: {val}")
