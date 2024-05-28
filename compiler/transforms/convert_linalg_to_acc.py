import sys
from dataclasses import dataclass, field

from xdsl.dialects import builtin, func, linalg, scf
from xdsl.ir import Block, MLContext, Operation, OpResult, Region, SSAValue, Use
from xdsl.parser import StringAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.accelerators.registry import AcceleratorRegistry
from compiler.dialects import acc


@dataclass
class ConvertLinalgToAcceleratorPattern(RewritePattern):
    """
    This pattern converts linalg generic ops that have been annotated
    with library_call.data = "snax_hwpe_mult" to the acc2 dialect.

    Eventually it should be converted to a generic pattern that handles
    all the different accelerators that snax has to offer. But not yet.
    """

    module: builtin.ModuleOp

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter, /):
        if op.library_call is None:
            return

        library_call_name = op.library_call.data

        acc_reg = AcceleratorRegistry()
        acc_names = acc_reg.get_names()
        if library_call_name not in acc_names:
            return

        # Lookup the accelerator interface based on the library_call
        _, acc_info = acc_reg.lookup_acc_info(
            StringAttr(library_call_name), self.module
        )
        rewriter.replace_matched_op(acc_info().convert_to_acc_ops(op))


@dataclass
class ConnectStatesThroughControlFlowPattern(RewritePattern):
    """
    This pass walks the control flow path of a function body and connects
    all the `acc2.setup()` ops together so that later analysis passes
    can infer where the state comes from.

    It currently handles scf.if and scf.for, but not more.
    """

    walked_funcs: set[str] = field(default_factory=set)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter, /):
        if func_op.sym_name.data in self.walked_funcs:
            return
        self.walked_funcs.add(func_op.sym_name.data)
        _weave_states_in_region(func_op, {}, rewriter)


def _weave_states_in_region(
    container: Region | Operation, state: dict[str, SSAValue], rewriter: PatternRewriter
) -> dict[str, SSAValue]:
    """
    Walks over a region or operation to "weave" accelerator setup states together.

    Takes a dictionary containing the SSA value carrying the current state at the start
    of the block for each accelerator, and *mutates* this dictionary to contain the SSA
    value carrying the accelerator state at the end of the block. This dictionary is
    also returned.

    Applies itself iteratively over nested regions inside of ops in this region, when
    it recognises the operations.

    Currently supports weaving through:
    - if/else blocks
    - for loops

    Assumes arith, memref, linalg, test and snax dialect ops can't modify the accelerator
    state.

    Also knows about some of the control flow dialects and terminator ops
    (they don't impact state).

    Assumes all other ops reset the accelerator state.
    """
    if isinstance(container, Operation):
        regions = container.regions
    else:
        regions = [container]

    for region in regions:
        for block in region.blocks:
            for op in block.ops:
                # handle acc.setup ops:
                if isinstance(op, acc.SetupOp):
                    accel = op.accelerator.data
                    if accel in state and op.in_state != state[accel]:
                        new_op = acc.SetupOp(
                            op.values,
                            op.param_names,
                            op.accelerator,
                            state[accel],
                        )
                        rewriter.replace_op(op, new_op)
                        op = new_op
                    state[accel] = op.out_state
                # special case for scf.if ops
                elif isinstance(op, scf.If):
                    # grab the computed state for both sides:
                    if_state = _weave_states_in_region(
                        op.true_region, state.copy(), rewriter
                    )
                    else_state = _weave_states_in_region(
                        op.false_region, state.copy(), rewriter
                    )

                    # calculate the delta:
                    delta = calc_if_state_delta(state, if_state, else_state)
                    # no delta = nothing to do
                    if not delta:
                        continue

                    # grab a list of added return vals for both branches
                    new_vals: tuple[tuple[SSAValue, ...], ...] = (
                        tuple(if_val for if_val, _ in delta.values()),
                        tuple(else_val for _, else_val in delta.values()),
                    )
                    # for each branch, rewrite the yield to return the new state
                    for branch, added_vals in zip(op.regions, new_vals):
                        assert isinstance(branch, Region)
                        if not branch.blocks:
                            branch.add_block(Block([scf.Yield(*added_vals)]))
                        else:
                            assert (
                                branch.block.last_op is not None
                            )  # we know there is a yield op
                            rewriter.replace_op(
                                branch.block.last_op,
                                scf.Yield(*branch.block.last_op.operands, *added_vals),
                            )
                    # then, insert a new if with additional return values:
                    num_scf_results = len(op.results)
                    rewriter.replace_op(
                        op,
                        new_if := scf.If(
                            op.cond,
                            [val.type for val in (*op.results, *new_vals[0])],
                            op.detach_region(op.regions[0]),
                            # is fine because previous line removes a region
                            op.detach_region(op.regions[0]),
                        ),
                        [new_if.results[i] for i in range(num_scf_results)],
                    )
                    # update state:
                    for res in new_if.results[num_scf_results:]:
                        assert isinstance(res.type, acc.StateType)
                        state[res.type.accelerator.data] = res
                # a for loop necessitates us to introduce a loop-carried variable
                # that carries the state through the loop
                elif isinstance(op, scf.For):
                    # create a state dictionary with TemporaryPlaceholderSSAValue so that we can track which new uses
                    # were introduced on this version of the state dictionary specifically:
                    mock_state = {
                        accel_name: TemporaryPlaceholderSSAValue(ssa_val)
                        for accel_name, ssa_val in state.items()
                    }

                    # go through the for loop body and weave states through that (this is a recursive call):
                    after_for_state = _weave_states_in_region(
                        op.body, mock_state.copy(), rewriter
                    )

                    # check which states got new uses:
                    # no state change in loop => nothing to do
                    if all(not val.new_uses() for val in mock_state.values()):
                        continue

                    # create a list of all accelerator names that were updated. The order of this list defines
                    # the order of all the other lists.
                    updated_accelerators = [
                        acc_name
                        for acc_name, new_state in after_for_state.items()
                        if mock_state.get(acc_name) != new_state
                    ]

                    # insert empty setup ops for all setups that don't have a state before the loop
                    for acc_name in updated_accelerators:
                        if acc_name not in state:
                            # create empty setup op
                            empty_setup = acc.SetupOp([], [], acc_name)
                            # insert op before the scf.for
                            rewriter.insert_op_before(empty_setup, op)
                            # register it as an inpup
                            state[acc_name] = empty_setup.out_state

                    # get a list of all initial states of accelerators that were changed int the loop.
                    input_states: list[SSAValue] = [
                        state[acc_name] for acc_name in updated_accelerators
                    ]

                    # grab the new states that are changed after the loop body executed.
                    new_states: list[SSAValue] = [
                        after_for_state[acc_name] for acc_name in updated_accelerators
                    ]

                    # add the input states as initial loop-carried states
                    op.operands = (*op.operands, *input_states)

                    # add changed states as yield ops in the loop
                    yield_op = op.body.block.last_op
                    assert isinstance(yield_op, scf.Yield)
                    yield_op.operands = (*yield_op.operands, *new_states)

                    # create the loop-carried block arguments
                    block_args = [
                        rewriter.insert_block_argument(
                            op.body.block, len(op.body.block.args), typ
                        )
                        for typ in (new_state.type for new_state in new_states)
                    ]

                    # this is only sound because the order of block_args is the same as the order of new_states,
                    # which is the same as the order of updated_accelerators.
                    for acc_name, block_arg in zip(
                        updated_accelerators, block_args, strict=True
                    ):
                        # we expect each state to have just one new use introduced inside the loop (the next setup op).
                        # future passes may require us to do something else, so I didn't want to assert here.
                        # this is sort of a sanity check for myself during testing.
                        if len(mock_state[acc_name].new_uses()) != 1:
                            print(
                                f"Unusual number of uses for acc {acc_name} in loop {op}..."
                            )
                        mock_state[acc_name].replace_new_uses_by(block_arg)
                        # also update state

                    # create new OpResults for the newly introduced loop-carried variables
                    # note that Mathieu is very unhappy about this.
                    new_results = [
                        OpResult(state_val.type, op, len(op.results))
                        for state_val in new_states
                    ]
                    # add them to the op
                    # Mathieu did not approve (he would prefer we create a new op, but that's also a lot of effort).
                    # I (Anton) thinks this is fine because it works.
                    op.results = (*op.results, *new_results)

                    # update states
                    for acc_name, result in zip(updated_accelerators, new_results):
                        state[acc_name] = result

                # calling another function invalidates all states
                # we can't reason about other functions as of now, and
                # adding support for that is out of scope for now.
                elif isinstance(op, func.Call):
                    state.clear()
                # arith, memref, linalg and test ops are not relevant to
                # accelerator setup, so we can skip them
                elif op.dialect_name() in (
                    "arith",
                    "memref",
                    "linalg",
                    "test",
                    "acc2",
                    "snax",
                ):
                    continue
                # these ops are specifically whitelisted:
                elif isinstance(op, func.Return | scf.Yield):
                    continue
                # for every other operation, raise a warning and just assume the worst
                else:
                    state.clear()
                    print(
                        f'[convert-linalg-to-acc] Unknown operation "{op.name}", '
                        f"assuming all side effects and resetting states.",
                        file=sys.stderr,
                    )
    return state


def calc_if_state_delta(
    old_state: dict[str, SSAValue],
    if_state: dict[str, SSAValue],
    else_state: dict[str, SSAValue],
) -> dict[str, tuple[SSAValue, SSAValue]]:
    """
    Given three state dictionaries (mapping accelerator names
    to SSA vals containing their state, return a new dict that:

    - Contains tuples (if_branch_val, else_branch_val)
    - For all accelerators whose state val changed in *at least
      one* of the branches
    - And for all accelerator states that got introduced in *both*
      branches
    """
    new_state: dict[str, tuple[SSAValue, SSAValue]] = {}

    # for every key in the old state, find out if it changed
    for k in old_state:
        # get new vals (or None if dropped)
        new_vals = (
            if_state.pop(k, None),
            else_state.pop(k, None),
        )
        # drop val if it is invalidated on one side
        if any(v is None for v in new_vals):
            continue
        # if no val changed
        if all(v == old_state[k] for v in new_vals):
            continue
        # pyright can't infer that we actually checked the argument type:
        new_state[k] = new_vals  # pyright: ignore[reportArgumentType]

    # check for states that are present in both branches
    for k in if_state:
        if k not in else_state:
            continue
        # add them to the new dict
        new_state[k] = (if_state[k], else_state[k])

    return new_state


class ConvertLinalgToAccPass(ModulePass):
    name = "convert-linalg-to-acc"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ConvertLinalgToAcceleratorPattern(op)).rewrite_module(op)
        # run these strictly sequentially, otherwise stuff breaks
        PatternRewriteWalker(ConnectStatesThroughControlFlowPattern()).rewrite_module(
            op
        )


class TraceStatesPass(ModulePass):
    """
    standalone version of state tracing for testing purposes
    """

    name = "accfg-trace-states"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ConnectStatesThroughControlFlowPattern()).rewrite_module(
            op
        )


@dataclass(init=False, eq=False)
class TemporaryPlaceholderSSAValue(SSAValue):
    """
    This value can be used temporarily in place of a normal SSA value, as long as all the places where it has been used
    are replaced by a "real" SSA value at the end of the rewrite.

    The idea is to pass this in the state of the _weave_states_in_region function, and then later inspect the states
    to see which ones were actually used in the body. This is a bit hacky. But it works and is kinda useful.
    """

    _wraps: SSAValue

    def __init__(self, val: SSAValue):
        super().__init__(val.type)
        self.uses.update(val.uses)
        self._wraps = val

    @property
    def owner(self) -> Operation | Block:
        return self._wraps.owner

    def new_uses(self) -> set[Use]:
        """
        Return the uses that were added to this value that are not present on the wrapped value.
        """
        return self.uses - self._wraps.uses

    def replace_new_uses_by(self, value: SSAValue) -> None:
        for use in self.new_uses().copy():
            use.operation.operands[use.index] = value
        # carry over name if possible
        if value.name_hint is None:
            value.name_hint = self._wraps.name_hint
        assert len(self.new_uses()) == 0, "unexpected error in xdsl"
