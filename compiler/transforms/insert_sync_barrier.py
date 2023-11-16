from xdsl.dialects import builtin
from xdsl.ir import MLContext
from xdsl.passes import ModulePass


class InsertSyncBarrier(ModulePass):
    name = "insert-sync-barrier"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        pass
