from xdsl.dialects import arith, builtin, func, llvm
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable

from compiler.dialects import snax


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

    Awaiting an llvm.inline_asm snitch runtime, implement the snax.allocs
    through C interfacing with function calls. The function call returns a
    pointer to the allocated memory. Aligned allocation is not supported yet,
    and the argument will be ignored. The pass is only implemented for L1 (TCDM)
    memory space allocations.

    In this pass we must also initialize the llvm struct with the correct contents
    for now we only populate pointer, aligned_pointer, offset and shapes.
    The strides are not pupulated because they are not used, and often not available.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, alloc_op: snax.Alloc, rewriter: PatternRewriter):
        ## only supoorting L1 allocation for now
        if not (
            isinstance(alloc_op.memory_space, builtin.IntegerAttr)
            and alloc_op.memory_space.value.data == 1
        ):
            return

        def dense_array(pos):
            return builtin.DenseArrayBase.create_dense_int_or_index(builtin.i64, pos)

        ops_to_insert = []

        func_call = func.Call(
            "snax_alloc_l1", [alloc_op.size], [llvm.LLVMPointerType.opaque()]
        )
        ops_to_insert.append(func_call)

        llvm_struct = llvm.UndefOp(alloc_op.result.type)
        ops_to_insert.append(llvm_struct)

        llvm_struct = llvm.InsertValueOp(
            dense_array([0]), llvm_struct.res, func_call.res[0]
        )
        ops_to_insert.append(llvm_struct)

        llvm_struct = llvm.InsertValueOp(
            dense_array([1]), llvm_struct.res, func_call.res[0]
        )
        ops_to_insert.append(llvm_struct)

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
            [builtin.IndexType()],
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
