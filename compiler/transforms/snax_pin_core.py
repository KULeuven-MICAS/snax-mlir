from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import builtin, func
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable
from xdsl.transforms.experimental.function_constant_pinning import (
    FunctionConstantPinning,
)


class RemoveExternalFunc(RewritePattern):
    """
    RewritePattern that temporarily removes all external function calls.
    It keeps track of all removed functions in its removed_function_calls variable.
    """

    removed_function_calls: list[Operation] = []

    @op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter):
        if func_op.sym_visibility == builtin.StringAttr("private"):
            self.removed_function_calls.append(func_op)
            rewriter.replace_matched_op([])


@dataclass
class ReinsertExternalFunc(RewritePattern):
    """
    RewritePattern that reinserts external function calls in "removed_function_calls".
    """

    removed_function_calls: list[Operation]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, module_op: builtin.ModuleOp, _: PatternRewriter):
        for op in self.removed_function_calls:
            SymbolTable.insert_or_update(module_op, op)


class SNAXCorePinningPass(ModulePass):
    """
    Pins functions to cores by using the experimental function_constant_pinning pass,
    can heavily benefit from inlining and canonicalization.
    It temporarily removes all external function declarations before applying FunctionConstantPinning,
    then it puts all external function calls back in place.

    This pass is meant to be used in conjunction with -p dispatch-regions{pin_to_constants=True}
    """

    name = "snax-pin-core"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        function_call_remover = RemoveExternalFunc()
        PatternRewriteWalker(function_call_remover).rewrite_module(op)
        PatternRewriteWalker(FunctionConstantPinning()).rewrite_module(op)
        PatternRewriteWalker(
            ReinsertExternalFunc(function_call_remover.removed_function_calls)
        ).rewrite_module(op)
