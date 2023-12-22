from xdsl.dialects import arith, builtin, func, memref
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable


class InsertFunctionCalls(RewritePattern):
    """
    Looks for memref copy operations and insert a snitch 1d dma call
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.CopyOp, rewriter: PatternRewriter):
        # If memref has rank > 1, it is not supported for now
        if not isinstance(op.source.type, memref.MemRefType) and not isinstance(
            op.destination.type, memref.MemRefType
        ):
            return
        if (
            op.source.type.get_num_dims() != 1
            or op.destination.type.get_num_dims() != 1
        ):
            return

        # Extract size information
        zero_const = arith.Constant.from_int_and_width(0, builtin.IndexType())
        dim_op = memref.Dim.from_source_and_index(op.source, zero_const.result)

        # Extract source and destination pointers
        source_ptr_op = memref.ExtractAlignedPointerAsIndexOp.get(op.source)
        dest_ptr_op = memref.ExtractAlignedPointerAsIndexOp.get(op.destination)

        # Make function call
        func_call = func.Call(
            "snax_dma_1d_transfer",
            [source_ptr_op.aligned_pointer, dest_ptr_op.aligned_pointer, dim_op.result],
            [],
        )

        # Replace op with function call
        rewriter.insert_op_before_matched_op(
            [zero_const, dim_op, source_ptr_op, dest_ptr_op]
        )
        rewriter.replace_op(op, func_call)


class SNAXCopyToDMA(ModulePass):
    """
    This pass translates memref copies to snitch DMA calls.
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
