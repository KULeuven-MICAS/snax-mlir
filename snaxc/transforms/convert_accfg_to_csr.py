from abc import ABC
from dataclasses import dataclass
from functools import cache

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from snaxc.accelerators import AccContext
from snaxc.accelerators.accelerator import Accelerator
from snaxc.dialects import accfg


@dataclass
class LowerAccfgBasePattern(RewritePattern, ABC):
    """
    Base class for the accfg dialect lowerings.

    Wraps some common logic to get handles to accelerator ops inside the module.
    """

    module: builtin.ModuleOp
    ctx: AccContext

    @cache
    def get_acc(self, accelerator_str: str) -> tuple[accfg.AcceleratorOp, Accelerator]:
        return self.ctx.get_acc_op_from_module(accelerator_str, self.module)

    def __hash__(self):
        return id(self)


class LowerAccfgSetupToCsr(LowerAccfgBasePattern):
    """
    Convert setup ops to a series of CSR sets that set each field to the given value.

    Looks up the csr addresses of the value fields by getting the `accfg.accelerator`
    operation from the module op.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: accfg.SetupOp, rewriter: PatternRewriter, /):
        acc_op, acc_info = self.get_acc(op.get_acc_name())
        # grab a dict that translates field names to CSR addresses:
        # emit the llvm assembly code to set csr values:
        rewriter.replace_matched_op(
            acc_info.lower_acc_setup(op, acc_op),
            [None],
            safe_erase=False,
        )


class LowerAccfgLaunchToCsr(LowerAccfgBasePattern):
    """
    Convert launch ops to a single `csr_set $launch_addr, 1`
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: accfg.LaunchOp, rewriter: PatternRewriter, /):
        assert isinstance(op.state.type, accfg.StateType)
        acc_op, acc_info = self.get_acc(op.get_acc_name())
        # acc_op, acc_info = self.get_acc(op.state.type.accelerator.data)
        # insert an op that sets the launch CSR to 1
        rewriter.replace_matched_op(
            acc_info.lower_acc_launch(op, acc_op),
            [op.state],
            safe_erase=False,
        )


class LowerAccfgAwaitToCsr(LowerAccfgBasePattern):
    """
    Lower await ops to a set of assembly that lowers to a buffer.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: accfg.AwaitOp, rewriter: PatternRewriter, /):
        acc_op, acc_info = self.get_acc(op.get_acc_name())

        # emit a snax_hwpe-style barrier
        rewriter.replace_matched_op(
            acc_info.lower_acc_await(acc_op),
            safe_erase=False,
        )


class DeleteAllStates(RewritePattern):
    """
    This pattern deletes all remaining SSA values that are of `accfg.state` type
    from any remaining operations.

    This is done to un-weave the `accfg.state` variables that were inserted into
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
        if any(isinstance(operand.type, accfg.StateType) for operand in op.operands):
            # use the generic creation interface to clone the op but with fewer
            # operands:
            new_op = op.__class__.create(
                operands=[operand for operand in op.operands if not isinstance(operand.type, accfg.StateType)],
                result_types=[res.type for res in op.results],
                properties=op.properties,
                attributes=op.attributes,
                successors=op.successors,
                regions=[reg.clone() for reg in op.regions],
            )
            # replace the op
            rewriter.replace_op(op, new_op, safe_erase=False)
            op = new_op

        # then we check if any of the results are of the offending type
        if any(isinstance(result.type, accfg.StateType) for result in op.results):
            # and again, clone the op but remove the results of the offending type
            new_op = op.__class__.create(
                operands=op.operands,
                result_types=[res.type for res in op.results if not isinstance(res.type, accfg.StateType)],
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
                (None if isinstance(res.type, accfg.StateType) else new_ops_results.pop(0)) for res in op.results
            ]
            # and then we replace the offending operation
            rewriter.replace_op(op, new_op, new_results=replace_results_by, safe_erase=False)

        # also clean up all block arguments
        for region in op.regions:
            for block in region.blocks:
                for arg in block.args:
                    if isinstance(arg.type, accfg.StateType):
                        rewriter.erase_block_argument(arg, safe_erase=False)


class RemoveAcceleratorOps(RewritePattern):
    """
    Delete all accelerator ops after we lowered the setup ops
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: accfg.AcceleratorOp, rewriter: PatternRewriter, /):
        rewriter.erase_matched_op()


class ConvertAccfgToCsrPass(ModulePass):
    """
    Converts accfg dialect ops to series of SNAX-like csr sets.
    """

    name = "convert-accfg-to-csr"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # first lower all accfg ops and erase old SSA values
        assert isinstance(ctx, AccContext)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerAccfgSetupToCsr(op, ctx),
                    LowerAccfgLaunchToCsr(op, ctx),
                    LowerAccfgAwaitToCsr(op, ctx),
                ]
            ),
            walk_reverse=True,
        ).rewrite_module(op)

        # then we remove all the top-level accfg.accelerator operations from the module and erase the state variables
        PatternRewriteWalker(GreedyRewritePatternApplier([DeleteAllStates(), RemoveAcceleratorOps()])).rewrite_module(
            op
        )
