from dataclasses import dataclass, field

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, linalg, memref, tensor
from xdsl.ir import BlockArgument, Operation, OpResult
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.transforms.mlir_opt import MLIROptPass

from snaxc.dialects import snax
from snaxc.transforms.alloc_to_global import AllocToGlobal
from snaxc.transforms.convert_tosa_to_kernel import RescaleClampPattern
from snaxc.transforms.test.insert_debugs import InsertDebugStatements


class InsertStaticFunctionCall(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: builtin.ModuleOp, rewriter: PatternRewriter):
        """
        Insert a function with static arguments them to dynamic ones,
        calls the existing main function, and then casts the dynamic output
        to a static value. After inlining and canonicalization, everything is static.
        """

        # define types
        static_input = builtin.TensorType(builtin.IntegerType(8), (8, 640))
        dynamic_input = builtin.TensorType(builtin.IntegerType(8), (-1, 640))

        static_output = builtin.TensorType(builtin.IntegerType(8), (8, 640))
        dynamic_output = builtin.TensorType(builtin.IntegerType(8), (-1, 640))

        @Builder.implicit_region((static_input,))
        def func_region(args: tuple[BlockArgument, ...]):
            cast = tensor.CastOp(args[0], dynamic_input)
            func_result = func.CallOp("main", [cast], [dynamic_output])
            cast2 = tensor.CastOp(func_result, static_output)
            func.ReturnOp(cast2)

        main_func = func.FuncOp(
            "run_network",
            builtin.FunctionType.from_lists([static_input], [static_output]),
            region=func_region,
        )

        rewriter.insert_op(main_func, InsertPoint.at_end(op.body.block))


class DropOldFunction(RewritePattern):
    """
    After the function `main` has been inlined in the function
    `run_network`, the main function can be dropped
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        ## delete old main function which has been inlined
        if op.sym_name.data == "main":
            rewriter.erase_matched_op()


class RemoveZeroInits(RewritePattern):
    """
    These ops initialize memory to zero before doing a matmul.
    With the gemm accelerator we use, this is not necessary.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.FillOp, rewriter: PatternRewriter):
        # delete fill ops that fill with zero, the accelerator will init from zero by default
        if len(op.inputs) != 1:
            return
        if not isinstance((const_input := op.inputs[0]), OpResult):
            return
        if not isinstance((const_op := const_input.op), arith.ConstantOp):
            return
        if not isinstance((intattr := const_op.value), builtin.IntegerAttr):
            return
        if intattr.value.data != 0:
            return

        # erase the op
        rewriter._replace_all_uses_with(op.results[0], op.outputs[0])
        rewriter.erase_matched_op()


class RemoveTransposeConstants(RewritePattern):
    """
    This path finds linalg generic operations that transpose a constant.
    It then constant folds this operation by transforming the weight directly.
    """

    def transpose_tuple(self, array_tuple: tuple, cols: int, rows: int):
        # Transpose using list comprehension
        transposed_tuple = tuple(
            array_tuple[i + j * rows] for i in range(rows) for j in range(cols)
        )
        return transposed_tuple

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        # find transpose generics on constants and just do it directly

        # transpose op has only yield
        if not isinstance(op.body.block.first_op, linalg.YieldOp):
            return

        # check for transpose:
        if len(op.indexing_maps) != 2:
            return
        if op.indexing_maps.data[0].data != AffineMap.from_callable(
            lambda x, y: (y, x)
        ):
            return
        if op.indexing_maps.data[1].data != AffineMap.from_callable(
            lambda x, y: (x, y)
        ):
            return

        # is input constant?
        if not isinstance(op.inputs[0], OpResult):
            return
        if not isinstance((const_op := op.inputs[0].op), arith.ConstantOp):
            return
        if not isinstance((const_type := op.inputs[0].type), builtin.TensorType):
            return
        if not isinstance(
            (dense_attr := const_op.value), builtin.DenseIntOrFPElementsAttr
        ):
            return

        # transpose const op
        transposed_data = self.transpose_tuple(
            dense_attr.data.data, *const_type.get_shape()
        )
        transposed_dense_attr = builtin.DenseIntOrFPElementsAttr.create_dense_int(
            op.outputs[0].type, transposed_data
        )

        # create new const_op
        new_const_op = arith.ConstantOp(transposed_dense_attr, op.outputs[0].type)

        # insert new const operation
        rewriter.insert_op(new_const_op, InsertPoint.before(const_op))

        # replace uses of transform with new const op
        rewriter._replace_all_uses_with(op.results[0], new_const_op.results[0])

        # delete const op and linalg op
        rewriter.erase_matched_op()
        rewriter.erase_op(const_op)

        pass


class InsertMemoryClears(RewritePattern):
    """
    Until we have an established way of handling memory allocation,
    just clear all memory before every linalg.generic.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        dump = snax.ClearL1()
        rewriter.insert_op(dump, InsertPoint.before(op))


class OrganizeGetGlobals(RewritePattern):
    """
    Put getglobals right before their first use
    Otherwise they get allocated too early, and the stupid
    memory clears will fuck up this shit
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, getglobal: memref.GetGlobalOp, rewriter: PatternRewriter
    ):
        assert getglobal.parent
        for firstuser in getglobal.parent.walk():
            if firstuser in {x.operation for x in getglobal.memref.uses} and isinstance(
                firstuser, Operation
            ):
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
        arguments=(
            "-inline",
            "-cse",
            "-canonicalize",
            "-mlir-print-local-scope",
            "-mlir-print-op-generic",
        )
    )

    mlir_lowering_pass = MLIROptPass(
        arguments=(
            "--pass-pipeline=builtin.module("
            + "func.func(tosa-to-linalg-named, tosa-to-tensor,"
            + "tosa-to-scf, tosa-to-linalg, tosa-to-arith, canonicalize))",
            "--mlir-print-op-generic",
            "--mlir-print-local-scope",
            "--allow-unregistered-dialect",
        )
    )

    mlir_bufferization_pass = MLIROptPass(
        arguments=(
            "--test-linalg-transform-patterns=test-generalize-pad-tensor",
            "--linalg-generalize-named-ops",
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=bufferize-function-boundaries allow-return-allocs-from-loops"
            + " function-boundary-type-conversion=identity-layout-map",
            "--mlir-print-op-generic",
            "--mlir-print-local-scope",
            "--allow-unregistered-dialect",
        )
    )

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            InsertStaticFunctionCall(), apply_recursively=False
        ).rewrite_module(op)
        self.mlir_inliner_pass.apply(ctx, op)
        PatternRewriteWalker(DropOldFunction()).rewrite_module(op)
        PatternRewriteWalker(RescaleClampPattern()).rewrite_module(op)
        self.mlir_lowering_pass.apply(ctx, op)
        PatternRewriteWalker(RemoveZeroInits(), apply_recursively=False).rewrite_module(
            op
        )
        PatternRewriteWalker(
            RemoveTransposeConstants(), apply_recursively=False
        ).rewrite_module(op)
        self.mlir_bufferization_pass.apply(ctx, op)
        PatternRewriteWalker(AllocToGlobal()).rewrite_module(op)
        PatternRewriteWalker(
            InsertMemoryClears(), apply_recursively=False
        ).rewrite_module(op)
        PatternRewriteWalker(
            InsertDebugStatements(), apply_recursively=False
        ).rewrite_module(op)
        PatternRewriteWalker(
            OrganizeGetGlobals(), apply_recursively=False
        ).rewrite_module(op)
