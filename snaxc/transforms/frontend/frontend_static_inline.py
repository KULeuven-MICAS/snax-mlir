from dataclasses import dataclass, field
from typing import cast

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import builtin, func, tensor
from xdsl.ir import Attribute, BlockArgument
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
)
from xdsl.rewriter import InsertPoint
from xdsl.transforms.mlir_opt import MLIROptPass
from xdsl.utils.hints import isa


@dataclass(frozen=True)
class StaticInlinePass(ModulePass):
    name = "frontend-static-inline"

    value: int
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

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # Inline first op of the module, that must be a function:
        if not isinstance(func_op := op.body.block.first_op, func.FuncOp):
            return
        rewriter = PatternRewriter(func_op)

        # Insert static function call
        # define types
        dynamic_inputs = [input for input in func_op.function_type.inputs]
        dynamic_outputs = [output for output in func_op.function_type.outputs]

        static_inputs: list[builtin.TensorType] = []
        static_outputs: list[builtin.TensorType] = []

        for dynamic_input in dynamic_inputs:
            assert isa(dynamic_input, builtin.TensorType[Attribute])
            new_shape = [self.value if x == builtin.DYNAMIC_INDEX else x for x in dynamic_input.get_shape()]
            static_inputs.append(builtin.TensorType(dynamic_input.get_element_type(), new_shape))
        dynamic_inputs = cast(list[builtin.TensorType[Attribute]], dynamic_inputs)

        for dynamic_output in dynamic_outputs:
            assert isa(dynamic_output, builtin.TensorType[Attribute])
            new_shape = [self.value if x == builtin.DYNAMIC_INDEX else x for x in dynamic_output.get_shape()]
            static_outputs.append(builtin.TensorType(dynamic_output.get_element_type(), new_shape))
        dynamic_outputs = cast(list[builtin.TensorType[Attribute]], dynamic_outputs)

        @Builder.implicit_region(static_inputs)
        def func_region(args: tuple[BlockArgument, ...]):
            casts: list[tensor.CastOp] = []
            for i, input in enumerate(dynamic_inputs):
                cast = tensor.CastOp(args[i], input)
                casts.append(cast)
            func_result = func.CallOp("main", casts, dynamic_outputs)
            casts = []
            for i, output in enumerate(static_outputs):
                cast = tensor.CastOp(func_result.res[i], output)
                casts.append(cast)
            func.ReturnOp(*casts)

        main_func = func.FuncOp(
            "run_network",
            builtin.FunctionType.from_lists(static_inputs, static_outputs),
            region=func_region,
        )

        rewriter.insert_op(main_func, InsertPoint.at_end(op.body.block))

        # Let MLIR do the inlining:
        self.mlir_inliner_pass.apply(ctx, op)

        # And drop the function with dynamic arguments:
        assert isinstance(func_op := op.body.block.first_op, func.FuncOp)
        rewriter.erase_op(func_op)
