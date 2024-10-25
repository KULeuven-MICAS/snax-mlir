from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.ir import OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.dialects import snax
from compiler.dialects.stream import StreamingRegionOp
from compiler.util.dispatching_rules import dispatch_to_compute, dispatch_to_dm


class InsertSyncBarrierRewriter(RewritePattern):
    """The algorithm used for this pass investigates these dependencies across
    cores to insert synchronization passes at the correct times. This is done by
    walking through every op in the IR. For every operand used by the operation,
    we check if it also used by an operation on another core. If this is the case,
    we must insert a synchronization barrier between the two."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, module: builtin.ModuleOp, rewriter: PatternRewriter):
        ops_to_sync = []

        ## walk the entire module in order
        for op in module.walk():
            ## if the current op must be synced, insert operation and clear the list
            if op in ops_to_sync:
                # insert op
                sync_op = snax.ClusterSyncOp()
                rewriter.insert_op_before(sync_op, op)

                # clear the list
                ops_to_sync = []

            if isinstance(op, snax.ClusterSyncOp):
                # synchronisation ok, clear list
                ops_to_sync = []

            # check all operands of current op
            for operand in [*op.operands, *op.results]:
                # check all ops that use the operand -> dependency with current op
                uses = operand.uses
                if isinstance(operand, OpResult) and isinstance(operand.op, snax.LayoutCast):
                    uses = uses.union(operand.op.source.uses)
                for op_use in uses:
                    # now check if op is dispatched to a specific core and the result
                    # is used on another core - if yes, there must be a synchronisation
                    # barrier between the two ops

                    if dispatch_to_dm(op) and not dispatch_to_dm(op_use.operation):
                        ops_to_sync.append(op_use.operation)

                    if dispatch_to_compute(op) and not dispatch_to_compute(
                        op_use.operation
                    ):
                        ops_to_sync.append(op_use.operation)


class InsertSyncBarrier(ModulePass):
    """This pass inserts  snax synchronisation barriers in a program.
    Synchronisation barriers are required when data is shared between
    multiple cores in a cluster, and the correct data flow of the
    program must be maintained."""

    name = "insert-sync-barrier"

    def apply(self, ctx: MLContext, module: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            InsertSyncBarrierRewriter(), apply_recursively=False
        ).rewrite_module(module)
