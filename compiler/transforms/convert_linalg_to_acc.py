import itertools

from dataclasses import dataclass, field
from typing import Sequence
from xdsl.ir import Operation, MLContext, SSAValue, Region, Block
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern, PatternRewriteWalker
from compiler.dialects import acc
from xdsl.dialects import linalg, builtin, arith, memref, func, scf


class AcceleratorConfig:
    name = "snax_hwpe_mult"

    fields = (
        'A', 'B', 'O', 'size'
    )

    def generate_vals(self, op: linalg.Generic) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        """
        Produce a `Sequence[Operation], SSAValue` tuple for each field that contains:

        - a list of operations that calculate the field value
        - a reference to the SSAValue containing the calculated field value
        """
        a, b, c = op.operands

        zero = arith.Constant.from_int_and_width(0, builtin.IndexType())
        dim = memref.Dim.from_source_and_index(a, zero)
        size = [zero, dim], dim.result

        ptrs = [
            (
                [ptr := memref.ExtractAlignedPointerAsIndexOp.get(ref)],
                ptr.aligned_pointer
            ) for ref in (a, b, c)
        ]

        return ptrs + [size]


@dataclass
class ConvertLinalgToAcceleratorPattern(RewritePattern):
    module: builtin.ModuleOp

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter, /):
        if op.library_call is None:
            return

        if op.library_call.data != 'snax_hwpe_mult':
            return

        # grab accelerator
        accelerator = AcceleratorConfig()

        # grab arguments
        args = accelerator.generate_vals(op)

        # insert ops to calculate arguments
        for new_ops, _ in args:
            rewriter.insert_op_before_matched_op(new_ops)

        # instantiate setup call
        rewriter.insert_op_before_matched_op(
            setup := acc.SetupOp([val for _, val in args], accelerator.fields, accelerator.name)
        )

        # launch
        rewriter.insert_op_before_matched_op(
            token := acc.LaunchOp(setup)
        )

        # await
        rewriter.replace_matched_op(
            acc.AwaitOp(token)
        )

@dataclass
class ConnectStatesThroughControlFlowPattern(RewritePattern):
    walked_funcs: set[str] = field(default_factory=set)
    @op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter, /):
        if func_op.sym_name.data in self.walked_funcs:
            return
        self.walked_funcs.add(func_op.sym_name.data)
        _walk(func_op, {}, rewriter)


def _walk(container: Region | Operation, state: dict[str, SSAValue], rewriter: PatternRewriter) -> dict[str, SSAValue]:
    if isinstance(container, Operation):
        regions = container.regions
    else:
        regions = [container]

    for region in regions:
        for block in region.blocks:
            for op in block.ops:
                # arith, memref, linalg and test ops are not relevant to accelerator setup
                if op.name.split('.')[0] in ('arith', 'memref', 'linalg', 'test'):
                    continue
                # handle acc2 dialect ops:
                elif op.name.startswith('acc2.'):
                    # the setup op is the only relevant one for now:
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
                    else:
                        pass
                elif isinstance(op, scf.If):
                    if_state = _walk(op.true_region, state.copy(), rewriter)
                    else_state = _walk(op.false_region, state.copy(), rewriter)

                    delta = calc_if_state_delta(state, if_state, else_state)
                    if not delta:
                        continue
                    # grab a list of added return vals for both if branch and else branch
                    new_vals: tuple[tuple[SSAValue, ...], ...] = (
                        tuple(if_val   for if_val, _   in delta.values()),
                        tuple(else_val for _, else_val in delta.values()),
                    )
                    # somehow rewrite this fucker
                    for branch, added_vals in zip(op.regions, new_vals):
                        assert isinstance(branch, Region)
                        if not branch.blocks:
                            branch.add_block(Block([scf.Yield(*added_vals)]))
                        else:
                            rewriter.replace_op(
                                branch.block.last_op,
                                scf.Yield(*branch.block.last_op.operands, *added_vals)
                            )

                    num_scf_results = len(op.results)
                    rewriter.replace_op(
                        op, new_if := scf.If(
                            op.cond,
                            [val.type for val in (*op.results, *new_vals[0])],
                            op.detach_region(op.regions[0]),
                            # trust me, this is correct. The previous line removes
                            # the block from the list
                            op.detach_region(op.regions[0])
                        ),
                        [new_if.results[i] for i in range(num_scf_results)]
                    )
                    # update state:
                    for res in new_if.results[num_scf_results:]:
                        state[res.type.accelerator.data] = res

                # calling another function invalidates all states
                elif isinstance(op, func.Call):
                    state.clear()
                elif isinstance(op, (func.Return, scf.Yield)):
                    continue
                else:
                    raise RuntimeError(f"What is a {op}?")
    return state


def calc_if_state_delta(old_state: dict[str, SSAValue], if_state: dict[str, SSAValue], else_state: dict[str, SSAValue]) -> dict[str, tuple[SSAValue, SSAValue]]:
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
        new_state[k] = new_vals

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
        PatternRewriteWalker(ConnectStatesThroughControlFlowPattern()).rewrite_module(op)





