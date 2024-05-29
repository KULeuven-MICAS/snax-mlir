from abc import ABC
from dataclasses import dataclass
from functools import cache

from xdsl.dialects import builtin
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

from compiler.accelerators.accelerator import Accelerator
from compiler.accelerators.registry import AcceleratorRegistry
from compiler.dialects import acc


@dataclass
class LowerAccBasePattern(RewritePattern, ABC):
    """
    Base class for the acc2 dialect lowerings.

    Wraps some common logic to get handles to accelerator ops inside the module.
    """

    module: builtin.ModuleOp

    @cache
    def get_acc(
        self, accelerator: StringAttr
    ) -> tuple[acc.AcceleratorOp, type[Accelerator]]:
        acc_op, acc_info = AcceleratorRegistry().lookup_acc_info(
            accelerator, self.module
        )
        return acc_op, acc_info

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
        acc_op, acc_info = self.get_acc(op.accelerator)
        # grab a dict that translates field names to CSR addresses:
        # emit the llvm assembly code to set csr values:
        rewriter.replace_matched_op(
            acc_info.lower_acc_setup(op, acc_op),
            safe_erase=False,
        )


class LowerAccLaunchToCsr(LowerAccBasePattern):
    """
    Convert launch ops to a single `csr_set $launch_addr, 1`
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.LaunchOp, rewriter: PatternRewriter, /):
        assert isinstance(op.state.type, acc.StateType)
        acc_op, acc_info = self.get_acc(op.state.type.accelerator)

        # insert an op that sets the launch CSR to 1
        rewriter.replace_matched_op(
            acc_info.lower_acc_launch(op, acc_op),
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
        acc_op, acc_info = self.get_acc(op.token.type.accelerator)

        # emit a snax_hwpe-style barrier
        rewriter.replace_matched_op(
            acc_info.lower_acc_await(acc_op),
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
        # bug in xDSL sometimes calls this pattern on already removed IR
        if op.parent_op() is None:
            return
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

        # also clean up all block arguments
        for region in op.regions:
            for block in region.blocks:
                for arg in block.args:
                    if isinstance(arg.type, acc.StateType):
                        rewriter.erase_block_argument(arg)

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
                ]
            ),
            walk_reverse=True,
        ).rewrite_module(op)

        # then we remove all the top-level acc2.accelerator operations from the module and erase the state variables
        PatternRewriteWalker(GreedyRewritePatternApplier([DeleteAllStates(), RemoveAcceleratorOps()])).rewrite_module(op)
