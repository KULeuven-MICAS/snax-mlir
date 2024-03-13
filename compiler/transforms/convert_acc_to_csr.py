from abc import ABC
from dataclasses import dataclass
from functools import cache

from xdsl.dialects import arith, builtin, llvm
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import MLContext, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable

from compiler.dialects import acc


@dataclass
class LowerAccBasePattern(RewritePattern, ABC):
    """
    Base class for the acc2 dialect lowerings.

    Wraps some common logic to get handles to accelerator ops inside the module.
    """

    module: builtin.ModuleOp

    @cache
    def get_acc(self, accelerator: StringAttr) -> acc.AcceleratorOp:
        """
        Get a reference to the accelerator
        """
        trait = self.module.get_trait(SymbolTable)
        assert trait is not None
        acc_op = trait.lookup_symbol(self.module, accelerator)
        if not isinstance(acc_op, acc.AcceleratorOp):
            raise RuntimeError(
                f"Invalid IR: no accelerator op for @{accelerator.data} found in module"
            )
        return acc_op

    def __hash__(self):
        return id(self)


class LowerAccSetupToCsr(LowerAccBasePattern):
    """
    Convert setup ops to a series of CSR sets that set each field to the given value.

    Looks up the csr addresses of the value fields by getting the `acc2.accelerator`
    operation from the module op.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.SetupOp, rewriter: PatternRewriter, /):
        # grab a dict that translates field names to CSR addresses:
        field_to_csr = dict(self.get_acc(op.accelerator).field_items())

        # emit the llvm assembly code to set csr values:
        for field, val in op.iter_params():
            addr = field_to_csr[field]
            rewriter.insert_op_before_matched_op(
                [
                    addr_val := arith.Constant(addr),
                    llvm.InlineAsmOp(
                        "csrw $0, $1",
                        "I, r",
                        [addr_val, val],
                        has_side_effects=True,
                    ),
                ]
            )
        # delete the old setup op
        rewriter.erase_matched_op(safe_erase=False)


class LowerAccLaunchToCsr(LowerAccBasePattern):
    """
    Convert launch ops to a single `csr_set $launch_addr, 1`
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.LaunchOp, rewriter: PatternRewriter, /):
        assert isinstance(op.state.type, acc.StateType)
        acc_op = self.get_acc(op.state.type.accelerator)

        # insert an op that sets the launch CSR to 1
        rewriter.replace_matched_op(
            [
                addr_val := arith.Constant(acc_op.launch_addr),
                val := arith.Constant(builtin.IntegerAttr.from_int_and_width(1, 5)),
                llvm.InlineAsmOp(
                    "csrw $0, $1",
                    # I = any 12 bit immediate, K = any 5 bit immediate
                    # The K allows LLVM to emit an `csrrwi` instruction,
                    # which has room for one 5 bit immediate only.
                    "I, K",
                    [addr_val, val],
                    has_side_effects=True,
                ),
            ],
            [op.state],
            safe_erase=False,
        )


class LowerAccAwaitToCsr(LowerAccBasePattern):
    """
    Lower await ops to a set of assembly that lowers to a buffer.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.AwaitOp, rewriter: PatternRewriter, /):
        assert isinstance(op.token.type, acc.StateType | acc.TokenType)
        acc_op = self.get_acc(op.token.type.accelerator)

        # TODO: this is a temporary solution that will be reworked eventually

        # emit a snax_hwpe-style barrier
        rewriter.replace_matched_op(
            [
                barrier_sw_barrier := arith.Constant(acc_op.barrier_sw_barrier),
                zero := arith.Constant(builtin.IntegerAttr.from_int_and_width(0, 5)),
                # FIXME: How to clobber a0?
                llvm.InlineAsmOp(
                    (
                        "\n"
                        "  csrr a0, $0\n"
                        "1:\n"
                        "  bnez a0, 1b\n"
                        "  csrwi 0x3c5, $1\n"  # Weird clear routine
                        "  nop\n"
                        "  nop\n"
                        "  nop\n"
                    ),
                    # I = any 12 bit immediate, K = any 5 bit immediate
                    # The K allows LLVM to emit an `csrrwi` instruction,
                    # which has room for one 5 bit immediate only.
                    "I,K",
                    [barrier_sw_barrier, zero],
                    has_side_effects=True,
                ),
            ],
            safe_erase=False,
        )


class DeleteAllStates(RewritePattern):
    """
    This pattern deletes all remaining SSA values that are of `acc.state` type
    from any remaining operations.

    This is done to un-weave the `acc2.state` variables that were inserted into
    control flow operations.
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        """
        This  method is implemented in two parts, because it felt easier to argue
        about operands separately from results. This shouldn't be a big problem,
        as most operations should have either arguments *or* results of the chosen
        type but rarely both.
        """
        # first rewrite operands:
        if any(isinstance(operand.type, acc.StateType) for operand in op.operands):
            # use the generic creation interface to clone the op but with fewer
            # operands:
            new_op = op.__class__.create(
                operands=[
                    operand
                    for operand in op.operands
                    if not isinstance(operand.type, acc.StateType)
                ],
                result_types=[res.type for res in op.results],
                properties=op.properties,
                attributes=op.attributes,
                successors=op.successors,
                regions=[reg.clone() for reg in op.regions],
            )
            # replace the op
            rewriter.replace_op(op, new_op)
            op = new_op

        # then we check if any of the results are of the offending type
        if any(isinstance(result.type, acc.StateType) for result in op.results):
            # and again, clone the op but remove the results of the offending type
            new_op = op.__class__.create(
                operands=op.operands,
                result_types=[
                    res.type
                    for res in op.results
                    if not isinstance(res.type, acc.StateType)
                ],
                properties=op.properties,
                attributes=op.attributes,
                successors=op.successors,
                regions=[op.detach_region(reg) for reg in tuple(op.regions)],
            )
            # now we need to tell the rewriter which results to "drop" from the
            # operation. In order to do that it expects a list[SSAValue | None]
            #  that maps the old results to either:
            #  - a new result to replace it, or
            #  - `None` to signify the erasure of the result var.
            # So we construct a new list that has that structure:

            # first we create a list of reverse-order results from the new op
            new_ops_results = list(new_op.results)
            # now we iterate the old results and use either:
            #  - `None` if the old result was erased, or
            #  - `new_results.pop(0)`, which is the next result of the new results
            replace_results_by = [
                None if isinstance(res.type, acc.StateType) else new_ops_results.pop(0)
                for res in op.results
            ]
            # and then we replace the offending operation
            rewriter.replace_op(op, new_op, new_results=replace_results_by)


class RemoveAcceleratorOps(RewritePattern):
    """
    Delete all accelerator ops after we lowered the setup ops
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.AcceleratorOp, rewriter: PatternRewriter, /):
        rewriter.erase_matched_op()


class ConvertAccToCsrPass(ModulePass):
    """
    Converts acc2 dialect ops to series of SNAX-like csr sets.
    """

    name = "convert-acc-to-csr"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        # first lower all acc2 ops and erase old SSA values
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerAccSetupToCsr(op),
                    LowerAccLaunchToCsr(op),
                    LowerAccAwaitToCsr(op),
                    DeleteAllStates(),
                ]
            ),
            walk_reverse=True,
        ).rewrite_module(op)

        # then we remove all the top-level acc2.accelerator operations from the module
        PatternRewriteWalker(RemoveAcceleratorOps()).rewrite_module(op)
