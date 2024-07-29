from xdsl.context import MLContext
from xdsl.dialects import builtin, func, linalg
from xdsl.dialects.memref import Cast, MemRefType
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable


class AddExternalFunc(RewritePattern):
    """
    Looks for hwpe function calls and adds an external
    func call to it for LLVM to link in
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, module: builtin.ModuleOp, rewriter: PatternRewriter):
        for op in module.walk():
            # Op must be linalg generic
            if not isinstance(op, linalg.Generic):
                continue

            if op.library_call is None:
                continue

            # all memref arguments must be cast to a new memref with a dynamic shape
            # to avoid type mismatching for multiple function calls with different
            # argument shapes
            cast_ops_to_insert = []
            operands = []
            for operand in op.operands:
                if isinstance(operand.type, MemRefType):
                    new_type = MemRefType(
                        operand.type.element_type,
                        [-1] * len(operand.type.shape),
                        operand.type.layout,
                        operand.type.memory_space,
                    )
                    cast = Cast.get(operand, new_type)
                    cast_ops_to_insert.append(cast)
                    operands.append(cast)
                else:
                    operands.append(operand)

            func_call = func.Call(op.library_call.data, operands, [])

            # Replace op with function call
            rewriter.replace_op(op, [*cast_ops_to_insert, func_call])

            # Insert external function definition

            # the memref arguments must be changed to dynamic shapes
            # for the input types
            input_types = [arg.type for arg in op.inputs]
            for i, input_type in enumerate(input_types):
                if isinstance(input_type, MemRefType):
                    input_types[i] = MemRefType(
                        input_type.element_type,
                        [-1] * len(input_type.shape),
                        input_type.layout,
                        input_type.memory_space,
                    )
            # do the same for output types
            output_types = [res.type for res in op.outputs]
            for i, output_type in enumerate(output_types):
                if isinstance(output_type, MemRefType):
                    output_types[i] = MemRefType(
                        output_type.element_type,
                        [-1] * len(output_type.shape),
                        output_type.layout,
                        output_type.memory_space,
                    )

            # both inputs and outputs are passed as inputs for the external function
            # (pass allocated output memref by reference to the external function)
            func_op = func.FuncOp.external(
                func_call.callee.string_value(),
                [*input_types, *output_types],
                [],
            )

            SymbolTable.insert_or_update(module, func_op)


class LinalgToLibraryCall(ModulePass):
    """
    This pass detects linalg operations with an external library call, and
    replaces them with a function call and definition.
    """

    name = "linalg-to-library-call"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(AddExternalFunc(), apply_recursively=False).rewrite_module(
            op
        )
