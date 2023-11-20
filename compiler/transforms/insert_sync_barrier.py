from xdsl.dialects import builtin, memref
from compiler.dialects import snax
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class InsertSyncBarrierRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, module: builtin.ModuleOp, rewriter: PatternRewriter):
        ops_to_sync = []

        ## walk the entire module in order
        for op in module.walk():
            if len(op.operands) + len(op.results) == 0:
                continue

            ## if the current op must be synced, insert operation and flush buffer
            if op in ops_to_sync:
                # insert op
                sync_op = snax.ClusterSyncOp()
                rewriter.insert_op_before(sync_op, op)

                # flush buffer
                ops_to_sync = []

            # check all operands of current op
            for operand in [*op.operands, *op.results]:
                # check all ops that use the operand -> dependency with current op
                for op_use in operand.uses:
                    # now check if op and op_use will be dispatched on
                    # different cores - if yes, there must be a synchronisation
                    # barrier between the two ops

                    # rules: all memref ops on dm core, all others on compute core
                    def xor(a, b):
                        return a and not b or not a and b

                    if xor(
                        type(op) in memref.MemRef.operations,
                        type(op_use.operation) in memref.MemRef.operations,
                    ):
                        # there must be a hw barrier synchronisation before the
                        # use op
                        ops_to_sync.append(op_use.operation)


class InsertSyncBarrier(ModulePass):
    name = "insert-sync-barrier"

    def apply(self, ctx: MLContext, module: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            InsertSyncBarrierRewriter(), apply_recursively=False
        ).rewrite_module(module)
