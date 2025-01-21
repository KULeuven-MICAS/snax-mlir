from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter

from compiler.dialects import snax
from compiler.util.dispatching_rules import dispatch_to_compute, dispatch_to_dm


class InsertSyncBarrier(ModulePass):
    """This pass inserts  snax synchronisation barriers in a program.
    Synchronisation barriers are required when data is shared between
    multiple cores in a cluster, and the correct data flow of the
    program must be maintained."""

    name = "insert-sync-barrier"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        rewriter = Rewriter()

        ops_to_sync = []

        ## walk the entire module in order
        for op_in_module in op.walk():
            ## if the current op must be synced, insert operation and clear the list
            if op_in_module in ops_to_sync:
                # insert op
                sync_op = snax.ClusterSyncOp()
                rewriter.insert_op(sync_op, InsertPoint.before(op_in_module))

                # clear the list
                ops_to_sync = []

            if isinstance(op_in_module, snax.ClusterSyncOp):
                # synchronisation ok, clear list
                ops_to_sync: list[Operation] = []

            # check all operands of current op
            for operand in [*op_in_module.operands, *op_in_module.results]:
                # check all ops that use the operand -> dependency with current op
                for op_use in operand.uses:
                    # now check if op is dispatched to a specific core and the result
                    # is used on another core - if yes, there must be a synchronisation
                    # barrier between the two ops

                    if dispatch_to_dm(op_in_module) and not dispatch_to_dm(
                        op_use.operation
                    ):
                        ops_to_sync.append(op_use.operation)

                    if dispatch_to_compute(op_in_module) and not dispatch_to_compute(
                        op_use.operation
                    ):
                        ops_to_sync.append(op_use.operation)
