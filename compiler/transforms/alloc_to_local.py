from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import builtin, memref
from xdsl.dialects.func import FuncOp, Return
from xdsl.parser import MemRefType
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.transforms.memref_to_snax import AllocOpRewrite
from compiler.util.snax_memory import L1


class AllocToLocal(RewritePattern):
    """
    Convert all memref allocs to be L1
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Alloc, rewriter: PatternRewriter):
        assert isinstance(memreftype := op.results[0].type, MemRefType)
        if memreftype.memory_space == L1:
            return
        new_type = builtin.MemRefType(memreftype.element_type, memreftype.get_shape(), memreftype.layout, L1)
        new_op = memref.Alloc([], [], new_type, op.alignment)
        for use in op.results[0].uses:
            if isinstance(use.operation, Return):
                assert isinstance(func_op := use.operation.parent.parent.parent, FuncOp)
                new_function_type = builtin.FunctionType.from_lists(func_op.function_type.inputs, [new_type])
                func_op.function_type = new_function_type
            if isinstance(use.operation, memref.CollapseShapeOp):
                old_collapse_type = use.operation.result.type
                new_collapse_type = builtin.MemRefType(old_collapse_type.element_type, old_collapse_type.get_shape(), memreftype.layout, L1)
                rewriter.replace_op(use.operation, memref.CollapseShapeOp(
                    operands=use.operation.operands,
                    result_types=[new_collapse_type],attributes=use.operation.attributes,properties=use.operation.properties
                ))
        rewriter.replace_matched_op(new_op)



@dataclass(frozen=True)
class AllocToLocalPass(ModulePass):
    name = "alloc-to-local"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(AllocToLocal()).rewrite_module(op)
        # PatternRewriteWalker(AllocOpRewrite()).rewrite_module(op)
