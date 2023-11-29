from xdsl.dialects import builtin, func, memref, arith
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
        # Exctract size information
        zero_const = arith.Constant.from_int_and_width(0, builtin.IndexType())
        seven_const = arith.Constant.from_int_and_width(7, builtin.IndexType())

        dim_op = memref.Dim.from_source_and_index(op.source, zero_const.result)
        source_ptr_op = memref.ExtractAlignedPointerAsIndexOp.get(op.source)
        dest_ptr_op = memref.ExtractAlignedPointerAsIndexOp.get(op.destination)
        func_call = func.Call(
            "snax_dma_1d_transfer",
            [source_ptr_op.aligned_pointer, dest_ptr_op.aligned_pointer, dim_op.result],
            [],
        )

        # Replace op with function call
        rewriter.insert_op_before_matched_op(
            [zero_const, seven_const, dim_op, source_ptr_op, dest_ptr_op]
        )
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
                [builtin.IndexType(), builtin.IndexType(), builtin.IndexType()],
                [],
            )
            SymbolTable.insert_or_update(op, func_decl)
