from dataclasses import dataclass, field

from xdsl.dialects import builtin, func, linalg, scf
from xdsl.ir import Block, MLContext, Operation, OpResult, Region, SSAValue
from xdsl.parser import StringAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.accelerators.registry import AcceleratorRegistry
from compiler.dialects import accfg
from compiler.dialects.snax_stream import StreamingRegionOp
from compiler.inference.helpers import (
    calc_if_state_delta,
    find_all_acc_names_in_region,
    find_existing_block_arg,
    has_accfg_effects,
)


@dataclass
class ConvertLinalgToAcceleratorPattern(RewritePattern):
    """
    This pattern converts linalg generic ops that have been annotated
    with library_call.data = "snax_hwpe_mult" to the accfg dialect.

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
class ConvertSnaxStreamToAcceleratorPattern(RewritePattern):
    """
    This pattern converts snax streaming region ops to the accfg dialect.
    """

    module: builtin.ModuleOp

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StreamingRegionOp, rewriter: PatternRewriter):
        _, acc_info = AcceleratorRegistry().lookup_acc_info(op.accelerator, self.module)

        rewriter.replace_matched_op(acc_info().convert_to_acc_ops(op))


@dataclass
class ConnectStatesThroughControlFlowPattern(RewritePattern):
    """
    This pass walks the control flow path of a function body and connects
    all the `accfg.setup()` ops together so that later analysis passes
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
                # handle accfg.setup ops:
                if isinstance(op, accfg.SetupOp):
                    accel = op.accelerator.data
                    if accel in state and op.in_state != state[accel]:
                        new_op = accfg.SetupOp(
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
                        assert isinstance(res.type, accfg.StateType)
                        state[res.type.accelerator.data] = res
                # a for loop necessitates us to introduce a loop-carried variable
                # that carries the state through the loop
                elif isinstance(op, scf.For):
                    # go through the for loop body find all accelerators that are touched
                    # the order of this tuple is important
                    updated_accelerators = tuple(
                        sorted(find_all_acc_names_in_region(op.body))
                    )

                    # check which states got new uses:
                    # no state change in loop => nothing to do
                    if not updated_accelerators:
                        continue

                    # insert empty setup ops for all setups that don't have a state before the loop
                    for acc_name in updated_accelerators:
                        if acc_name not in state:
                            # create empty setup op
                            empty_setup = accfg.SetupOp([], [], acc_name)
                            # insert op before the scf.for
                            rewriter.insert_op_before(empty_setup, op)
                            # register it as an input
                            state[acc_name] = empty_setup.out_state

                    # this is the state we are planning to pass to the weaving function inside the for body:
                    inner_state = state.copy()

                    # create or find the loop-carried block arguments we need to generate
                    # and populate the inner_state with them:
                    created_block_args = []
                    for accel in updated_accelerators:
                        arg = find_existing_block_arg(op.body.block, accel)
                        if arg is None:
                            arg = rewriter.insert_block_argument(
                                op.body.block,
                                len(op.body.block.args),
                                accfg.StateType(accel),
                            )
                            created_block_args.append(arg)
                        inner_state[accel] = arg

                    # weave vals with input states
                    after_for_state = _weave_states_in_region(
                        op.body, inner_state, rewriter
                    )

                    # get a list of all initial states of accelerators that were changed int the loop.
                    input_states: list[SSAValue] = [
                        state[acc_name]
                        for acc_name in updated_accelerators
                        if state[acc_name] not in op.operands
                    ]
                    # and add the input states as initial loop-carried states
                    op.operands = (*op.operands, *input_states)

                    # add changed states as yield ops in the loop
                    yield_op = op.body.block.last_op
                    assert isinstance(yield_op, scf.Yield)

                    # make sure we modify the for loop to add the new loop carried variables
                    for arg in created_block_args:
                        assert isinstance(arg.type, accfg.StateType)
                        acc_name = arg.type.accelerator.data
                        # extend the yield op to yield the state variable
                        yield_op.operands = (
                            *yield_op.operands,
                            after_for_state[acc_name],
                        )
                        # extend the op results to return the new state
                        new_result = OpResult(arg.type, op, len(op.results))
                        op.results = (*op.results, new_result)

                    # update states
                    for result in op.results:
                        if isinstance(result.type, accfg.StateType):
                            # update the state to reflect this
                            state[result.type.accelerator.data] = result

                # Check if the op has effects on accfg state
                elif has_accfg_effects(op):
                    state.clear()

    return state


class ConvertLinalgToAccPass(ModulePass):
    name = "convert-linalg-to-accfg"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ConvertLinalgToAcceleratorPattern(op)).rewrite_module(op)
        PatternRewriteWalker(ConvertSnaxStreamToAcceleratorPattern(op)).rewrite_module(
            op
        )
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
