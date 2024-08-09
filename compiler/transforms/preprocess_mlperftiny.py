from dataclasses import dataclass, field
from typing import cast

from xdsl.builder import Builder
from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, func, linalg, memref, tensor
from xdsl.ir import BlockArgument, OpResult, Operation
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable
from xdsl.transforms.mlir_opt import MLIROptPass

from compiler.dialects import snax
from compiler.util.kernel_type import KernelType


class InsertStaticFunctionCall(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: builtin.ModuleOp, rewriter: PatternRewriter):
        """
        Insert a function with static arguments them to dynamic ones,
        calls the existing main function, and then casts the dyanmic output
        to a static value. After inlining, everything is static then.
        """

        # define types
        static_input = builtin.TensorType(builtin.IntegerType(8), (8, 640))
        dynamic_input = builtin.TensorType(builtin.IntegerType(8), (-1, 640))

        static_output = builtin.TensorType(builtin.IntegerType(8), (8, 640))
        dynamic_output = builtin.TensorType(builtin.IntegerType(8), (-1, 640))

        @Builder.implicit_region((static_input,))
        def func_region(args: tuple[BlockArgument, ...]):
            cast = tensor.CastOp(args[0], dynamic_input)
            func_result = func.Call("main", [cast], [dynamic_output])
            cast2 = tensor.CastOp(func_result, static_output)
            func.Return(cast2)

        main_func = func.FuncOp(
            "run_network", builtin.FunctionType.from_lists([static_input], [static_output]), region=func_region
        )

        rewriter.insert_op(main_func, InsertPoint.at_end(op.body.block))


class DropOldFunction(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        ## delete old main function which has been inlined
        if op.sym_name.data == "main":
            rewriter.erase_matched_op()


class AllocToGlobal(RewritePattern):
    index: int = 0
    """
    Convert all static allocs to empty statically scheduled memref globals
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Alloc, rewriter: PatternRewriter):
        # get module op
        module_op = op
        while not isinstance(module_op, builtin.ModuleOp):
            assert module_op is not None
            module_op = module_op.parent_op()

        symbol_table = module_op.get_trait(SymbolTable)
        assert symbol_table

        # create global symbol name
        global_sym_name = "_static_const_" + str(self.index)
        self.index = self.index + 1

        # check if it is not in the symbol table
        assert SymbolTable.lookup_symbol(module_op, global_sym_name) is None

        # create global
        memref_global = memref.Global.get(
            builtin.StringAttr(global_sym_name), op.results[0].type, initial_value=builtin.UnitAttr()
        )
        SymbolTable.insert_or_update(module_op, memref_global)

        # remove all the deallocs
        deallocs = [
            user_op.operation for user_op in op.results[0].uses if isinstance(user_op.operation, memref.Dealloc)
        ]
        for dealloc in deallocs:
            rewriter.erase_op(dealloc)

        # replace op by get global
        memref_get = memref.GetGlobal(global_sym_name, op.results[0].type)

        rewriter.replace_matched_op(memref_get)

class RemoveZeroInits(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.FillOp, rewriter: PatternRewriter):
        # delete fill ops that fill with zero, the accelerator will init from zero by default
        if len(op.inputs) != 1:
            return
        if not isinstance((const_input := op.inputs[0]), OpResult):
            return
        if not isinstance((const_op := const_input.op), arith.Constant):
            return
        if not isinstance((intattr := const_op.value), builtin.IntegerAttr):
            return
        if intattr.value.data != 0:
            return

        # erase the op
        rewriter._replace_all_uses_with(op.results[0], op.outputs[0])
        rewriter.erase_matched_op()

class RemoveTransposeConstants(RewritePattern):

    def transpose_tuple(self, array_tuple: tuple, cols: int, rows: int):
        # Transpose using list comprehension
        transposed_tuple = tuple(array_tuple[i + j * rows] for i in range(rows) for j in range(cols))
        return transposed_tuple

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):

        # find transpose generics on constants and just do it directly
        kernel_type = KernelType.get_kernel(op)

        if kernel_type != KernelType.YIELD:
            return

        # check for transpose:
        if len(op.indexing_maps) != 2:
            return
        if op.indexing_maps.data[0].data != AffineMap.from_callable(lambda x, y: (y, x)):
            return
        if op.indexing_maps.data[1].data != AffineMap.from_callable(lambda x, y: (x, y)):
            return

        # is input constant?
        if not isinstance(op.inputs[0], OpResult):
            return
        if not isinstance((const_op := op.inputs[0].op), arith.Constant):
            return
        if not isinstance((const_type := op.inputs[0].type), builtin.TensorType):
            return
        if not isinstance((dense_attr := const_op.value), builtin.DenseIntOrFPElementsAttr):
            return

        # transpose const op
        transposed_data = self.transpose_tuple(dense_attr.data.data, *const_type.get_shape())
        transposed_dense_attr = builtin.DenseIntOrFPElementsAttr.create_dense_int(op.outputs[0].type, transposed_data)

        # create new const_op
        new_const_op = arith.Constant(transposed_dense_attr, op.outputs[0].type)

        # insert new const operation
        rewriter.insert_op(new_const_op, InsertPoint.before(const_op))

        # replace uses of transform with new const op
        rewriter._replace_all_uses_with(op.results[0], new_const_op.results[0])

        # delete const op and linalg op
        rewriter.erase_matched_op()
        rewriter.erase_op(const_op)

        pass

class RemoveIdioticClamps(RewritePattern):
    """
    Sometimes there is a clamping operation that clamps i8 values between -128 and 127
    This is completely idiotic. This pass removes that operation
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):

        kernel_type = KernelType.get_kernel(op)

        if kernel_type != KernelType.CLAMP:
            return

        assert op.body.block.last_op
        _, lower, upper = KernelType.parse_clamp(op.body.block.last_op.operands[0].op) #pyright: ignore


        if upper.result.op.value.value.data < 127:
            return

        if lower.result.op.value.value.data > -128:
            return

        if op.inputs[0].type.element_type.bitwidth != 8:
            return

        # otherwise, it is completely useless

        rewriter._replace_all_uses_with(op.results[0], op.inputs[0])
        rewriter.erase_matched_op()

        pass

class InsertMemoryDumps(RewritePattern):
    """
    Until we have an established way of handling memory allocation,
    just dump all memory after every linalg.generic.
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        dump = snax.DumpL1()
        rewriter.insert_op(dump, InsertPoint.after(op))

class InsertDebugStatements(RewritePattern):
    """
    Insert debugs :)
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        kernel_type = KernelType.get_kernel(op)

        if kernel_type == KernelType.QMAC:
            ## matmuls
            debug = snax.Debug(op.inputs[0], op.inputs[1], op.outputs[0], "gemm", "before")
            debug2 = snax.Debug(op.inputs[0], op.inputs[1], op.outputs[0], "gemm", "after")
            rewriter.insert_op(debug, InsertPoint.before(op))
            rewriter.insert_op(debug2, InsertPoint.after(op))

        if kernel_type == KernelType.ADD:
            ## bias add
            debug = snax.Debug(op.inputs[0], op.inputs[1], op.outputs[0], "bias", "before")
            debug2 = snax.Debug(op.inputs[0], op.inputs[1], op.outputs[0], "bias", "after")
            rewriter.insert_op(debug, InsertPoint.before(op))
            rewriter.insert_op(debug2, InsertPoint.after(op))

        if kernel_type == KernelType.RESCALE:
            ## rescale and clamp
            debug = snax.Debug(op.inputs[0], op.inputs[0], op.outputs[0], "simd", "before")
            debug2 = snax.Debug(op.inputs[0], op.inputs[0], op.outputs[0], "simd", "after")
            rewriter.insert_op(debug, InsertPoint.before(op))
            rewriter.insert_op(debug2, InsertPoint.after(op))

class OrganizeGetGlobals(RewritePattern):
    """
    Put getglobals right before their first use
    Otherwise they get allocated too early, and the stupid
    memory dumps will fuck up this shit
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, getglobal: memref.GetGlobal, rewriter: PatternRewriter):

        assert getglobal.parent
        for firstuser in getglobal.parent.walk():
            if firstuser in {x.operation for x in getglobal.memref.uses} and isinstance(firstuser, Operation):
                while firstuser.parent != getglobal.parent:
                    assert firstuser.parent
                    firstuser = firstuser.parent
                assert isinstance(firstuser, Operation)
                if firstuser.parent is getglobal.parent:
                    getglobal.detach()
                    return rewriter.insert_op(getglobal, InsertPoint.before(firstuser))




@dataclass(frozen=True)
class PreprocessMLPerfTiny(ModulePass):
    name = "preprocess-mlperftiny"

    executable: str = field(default="mlir-opt")
    generic: bool = field(default=True)

    mlir_inliner_pass = MLIROptPass(
        arguments=("-inline", "-cse", "-canonicalize", "-mlir-print-local-scope", "-mlir-print-op-generic")
    )
    mlir_bufferization_pass = MLIROptPass(
        arguments=(
            '--test-linalg-transform-patterns=test-generalize-pad-tensor',
            "--linalg-generalize-named-ops",
            "--empty-tensor-to-alloc-tensor",
            '--one-shot-bufferize=bufferize-function-boundaries allow-return-allocs'
            + ' function-boundary-type-conversion=identity-layout-map',
            "--mlir-print-op-generic",
            "--mlir-print-local-scope",
        )
    )

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        pass
        PatternRewriteWalker(RemoveZeroInits(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(RemoveTransposeConstants(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(RemoveIdioticClamps(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(InsertStaticFunctionCall(), apply_recursively=False).rewrite_module(op)
        self.mlir_inliner_pass.apply(ctx, op)
        PatternRewriteWalker(DropOldFunction()).rewrite_module(op)
        self.mlir_bufferization_pass.apply(ctx, op)
        PatternRewriteWalker(AllocToGlobal()).rewrite_module(op)
        PatternRewriteWalker(InsertMemoryDumps(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(InsertDebugStatements(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(OrganizeGetGlobals(), apply_recursively=False).rewrite_module(op)
