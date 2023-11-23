from xdsl.dialects import builtin, func, memref
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable


class InsertFunctionCalls(RewritePattern):
    """
    Looks for hwpe function calls and adds an external
    func call to it for LLVM to link in
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.CopyOp, rewriter: PatternRewriter):
        func_call = func.Call("snax_dma_1d_transfer", [op.source, op.destination], [])

        # Replace op with function call
        rewriter.replace_op(op, func_call)


class SNAXCopyToDMA(ModulePass):
    """
    This pass detects linalg operations with an external library call, and
    replaces them with a function call and definition.
    """

    name = "snax-copy-to-dma"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        contains_copies = any(
            isinstance(op_in_module, memref.CopyOp) for op_in_module in op.walk()
        )

        if contains_copies:
            PatternRewriteWalker(InsertFunctionCalls()).rewrite_module(op)
            func_decl = func.FuncOp.external(
                "snax_dma_1d_transfer",
                [
                    memref.MemRefType.from_element_type_and_shape(builtin.i32, [-1]),
                    memref.MemRefType.from_element_type_and_shape(builtin.i32, [-1]),
                ],
                [],
            )
            SymbolTable.insert_or_update(op, func_decl)
