from xdsl.context import Context
from xdsl.dialects import builtin, func
from xdsl.dialects.memref import DeallocOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable

from snaxc.dialects import snax


class InsertFunctionCall(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: snax.ClusterSyncOp, rewriter: PatternRewriter):
        """Swap cluster sync op with function call"""
        func_call = func.CallOp("snax_cluster_hw_barrier", [], [])
        rewriter.replace_matched_op(func_call)


class ClearL1ToFunc(RewritePattern):
    """Insert function call to clear l1"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, clear: snax.ClearL1, rewriter: PatternRewriter):
        func_call = func.CallOp("snax_clear_l1", [], [])
        func_decl = func.FuncOp.external("snax_clear_l1", [], [])

        # find module_op and insert func call
        module_op = clear
        while not isinstance(module_op, builtin.ModuleOp):
            assert (module_op := module_op.parent_op())
        SymbolTable.insert_or_update(module_op, func_decl)

        rewriter.replace_matched_op(func_call)


class EraseDeallocs(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DeallocOp, rewriter: PatternRewriter):
        rewriter.erase_matched_op()


class SNAXToFunc(ModulePass):
    name = "snax-to-func"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        contains_sync = any(isinstance(op_in_module, snax.ClusterSyncOp) for op_in_module in op.walk())

        if contains_sync:
            PatternRewriteWalker(InsertFunctionCall()).rewrite_module(op)
            func_op = func.FuncOp.external("snax_cluster_hw_barrier", [], [])
            SymbolTable.insert_or_update(op, func_op)

        PatternRewriteWalker(ClearL1ToFunc()).rewrite_module(op)
        PatternRewriteWalker(EraseDeallocs()).rewrite_module(op)
