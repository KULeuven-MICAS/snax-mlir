from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, func, llvm
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable

from compiler.dialects import snax
from compiler.util.snax_memory import L1


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


class AllocToFunc(RewritePattern):
    """Swap snax.alloc with function call

    This function implements the snax.allocs
    through C interfacing with function calls. The function call returns a
    pointer to the allocated memory. Aligned allocation is not supported yet,
    and the argument will be ignored. The pass is only implemented for L1 (TCDM)
    memory space allocations.

    In this pass we must also initialize the llvm struct with the correct contents
    for now we only populate pointer, aligned_pointer, offset and shapes.
    The strides are not populated because they are not used, and often not available.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, alloc_op: snax.Alloc, rewriter: PatternRewriter):
        ## only supporting L1 allocation for now
        if alloc_op.memory_space != L1:
            return

        def dense_array(pos):
            return builtin.DenseArrayBase.create_dense_int_or_index(builtin.i64, pos)

        ops_to_insert = []

        # create constant alignment op to pass on to the allocation function
        alignment_op = arith.Constant.from_int_and_width(
            alloc_op.alignment.value.data, builtin.IndexType()
        )
        ops_to_insert.append(alignment_op)

        # call allocation function with size and alignment operations
        func_call = func.Call(
            "snax_alloc_l1",
            [alloc_op.size, alignment_op],
            [llvm.LLVMPointerType.opaque()],
        )
        ops_to_insert.append(func_call)

        # the result of the function points to a struct containing 2 pointers:
        # the allocated pointer and the aligned pointer
        func_result_type = llvm.LLVMStructType.from_type_list(
            [
                llvm.LLVMPointerType.opaque(),  # pointer
                llvm.LLVMPointerType.opaque(),  # aligned_pointer
            ]
        )

        # load this struct
        func_result = llvm.LoadOp(func_call.res[0], func_result_type)
        ops_to_insert.append(func_result)

        # extract the allocated pointer and aligned pointer from alloc function call
        pointer_op = llvm.ExtractValueOp(
            dense_array([0]), func_result.results[0], llvm.LLVMPointerType.opaque()
        )
        aligned_pointer_op = llvm.ExtractValueOp(
            dense_array([1]), func_result.results[0], llvm.LLVMPointerType.opaque()
        )
        ops_to_insert.extend([pointer_op, aligned_pointer_op])

        # create the memref descriptor struct
        llvm_struct = llvm.UndefOp(alloc_op.result.type)
        ops_to_insert.append(llvm_struct)

        # insert pointer
        llvm_struct = llvm.InsertValueOp(
            dense_array([0]), llvm_struct.res, pointer_op.res
        )
        ops_to_insert.append(llvm_struct)

        # insert aligned pointer
        llvm_struct = llvm.InsertValueOp(
            dense_array([1]), llvm_struct.res, aligned_pointer_op.res
        )
        ops_to_insert.append(llvm_struct)

        # insert offset
        cst_zero = arith.Constant.from_int_and_width(0, builtin.i32)
        llvm_struct = llvm.InsertValueOp(dense_array([2]), llvm_struct.res, cst_zero)
        ops_to_insert.extend([cst_zero, llvm_struct])

        # use shape operands to populate shape of memref descriptor
        for i, shape_op in enumerate(alloc_op.shapes):
            if isinstance(shape_op.type, builtin.IndexType):
                # we must cast to integer for valid llvm op
                shape_op = builtin.UnrealizedConversionCastOp.get(
                    [shape_op], [builtin.i32]
                )
                ops_to_insert.append(shape_op)
            llvm_struct = llvm.InsertValueOp(
                dense_array([3, i]), llvm_struct.res, shape_op.results[0]
            )
            ops_to_insert.append(llvm_struct)

        module_op = alloc_op.get_toplevel_object()

        rewriter.replace_matched_op(ops_to_insert)

        func_op = func.FuncOp.external(
            "snax_alloc_l1",
            [builtin.IndexType()] * 2,
            [llvm.LLVMPointerType.opaque()],
        )

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

        PatternRewriteWalker(AllocToFunc()).rewrite_module(module)
