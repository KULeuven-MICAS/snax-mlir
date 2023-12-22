from xdsl.dialects import builtin, func, linalg
from xdsl.ir import MLContext
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

            func_call = func.Call(op.library_call.data, op.operands, [])

            # Replace op with function call
            rewriter.replace_op(op, func_call)

            # Insert external function definition
            func_op = func.FuncOp.external(
                func_call.callee.string_value(),
                [arg.type for arg in func_call.arguments],
                [res.type for res in func_call.results],
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
