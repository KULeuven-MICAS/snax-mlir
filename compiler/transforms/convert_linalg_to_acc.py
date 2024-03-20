import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

from xdsl.dialects import arith, builtin, func, linalg, memref, scf
from xdsl.ir import Block, MLContext, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable

from compiler.dialects import acc


class HWPEAcceleratorInfo:
    name = "snax_hwpe_mult"

    fields = ("A", "B", "O", "vector_length", "nr_iters", "mode")

    def generate_vals(
        self, op: linalg.Generic
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        """
        Produce a `Sequence[Operation], SSAValue` tuple for each field that contains:

        - a list of operations that calculate the field value
        - a reference to the SSAValue containing the calculated field value
        """
        a, b, c = op.operands

        zero = arith.Constant.from_int_and_width(0, builtin.IndexType())
        iters_one = arith.Constant.from_int_and_width(1, 32)
        mode_one = arith.Constant.from_int_and_width(1, 32)
        dim = memref.Dim.from_source_and_index(a, zero)
        dim_i32 = arith.IndexCastOp(dim, builtin.i32)
        vector_length = [zero, dim, dim_i32], dim_i32.result

        nr_iters = [iters_one], iters_one.result
        mode = [mode_one], mode_one.result

        ptrs = [
            (
                [
                    ptr := memref.ExtractAlignedPointerAsIndexOp.get(ref),
                    ptr_i32 := arith.IndexCastOp(ptr, builtin.i32),
                ],
                ptr_i32.result,
            )
            for ref in (a, b, c)
        ]

        return ptrs + [nr_iters] + [vector_length] + [mode]

    def generate_acc_op(self) -> acc.AcceleratorOp:
        """
        Return this accelerator op:

        "acc2.accelerator"() <{
            name            = @snax_hwpe_mult,
            fields          = {A=0x3d0, B=0x3d1, O=0x3d3, n_iters=0x3d4,
                               vector_length=0x3d5, mode=0x3d6},
            launch_addr     = 0x3c0,
            barrier = 0x3c3,
        }> : () -> ()
        """
        return acc.AcceleratorOp(
            self.name,
            {
                "A": 0x3D0,
                "B": 0x3D1,
                "O": 0x3D3,
                "vector_length": 0x3D4,
                "nr_iters": 0x3D5,
                "mode": 0x3D6,
            },
            0x3C0,
            0x3C3,
        )


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
        if op.library_call and op.library_call.data != "snax_hwpe_mult":
            return

        # grab accelerator
        accelerator = HWPEAcceleratorInfo()

        t = self.module.get_trait(SymbolTable)
        assert t is not None
        t.insert_or_update(self.module, accelerator.generate_acc_op())

        # grab arguments
        args = accelerator.generate_vals(op)

        # insert ops to calculate arguments
        for new_ops, _ in args:
            rewriter.insert_op_before_matched_op(new_ops)

        # instantiate setup call
        rewriter.insert_op_before_matched_op(
            setup := acc.SetupOp(
                [val for _, val in args], accelerator.fields, accelerator.name
            )
        )

        # launch
        rewriter.insert_op_before_matched_op(token := acc.LaunchOp(setup))

        # await
        rewriter.replace_matched_op(acc.AwaitOp(token))


@dataclass
class ConnectStatesThroughControlFlowPattern(RewritePattern):
    """
    This pass walks the control flow path of a function body and connects
    all the `acc2.setup()` ops together so that later analysis passes
    can infer where the state comes from.

    It currently handles scf.for, but not more.
    """

    walked_funcs: set[str] = field(default_factory=set)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter, /):
        if func_op.sym_name.data in self.walked_funcs:
            return
        self.walked_funcs.add(func_op.sym_name.data)
        _walk(func_op, {}, rewriter)


def _walk(
    container: Region | Operation, state: dict[str, SSAValue], rewriter: PatternRewriter
) -> dict[str, SSAValue]:
    """
    Walks over a region or operation to "weave" accelerator setup states
    together.

    Takes an input state, and returns an output state.
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
                    if_state = _walk(op.true_region, state.copy(), rewriter)
                    else_state = _walk(op.false_region, state.copy(), rewriter)

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
