from xdsl.dialects import builtin, func
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from compiler.dialects import snax
from xdsl.traits import SymbolTable


class InsertFunctionCall(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: snax.ClusterSyncOp, rewriter: PatternRewriter):
        """Swap cluster sync op with function call"""
        func_call = func.Call("snax_cluster_hw_barrier", [], [])
        rewriter.replace_matched_op(func_call)


class InsertFunctionDeclaration(RewritePattern):
    """Insert external function declarations of snax_cluster_hw_barrier"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, module_op: builtin.ModuleOp, rewriter: PatternRewriter):
        func_op = func.FuncOp.external("snax_cluster_hw_barrier", [], [])
        SymbolTable.insert_or_update(module_op, func_op)


class SNAXToFunc(ModulePass):
    name = "snax-to-func"

    def apply(self, ctx: MLContext, module: builtin.ModuleOp) -> None:
        contains_sync = any(
            isinstance(op_in_module, snax.ClusterSyncOp)
            for op_in_module in module.walk()
        )

        if contains_sync:
            PatternRewriteWalker(InsertFunctionCall()).rewrite_module(module)
            PatternRewriteWalker(InsertFunctionDeclaration()).rewrite_module(module)
