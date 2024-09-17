from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import builtin, linalg
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from compiler.dialects.test import debug


@dataclass(frozen=True)
class InsertDebugStatements(RewritePattern):
    """
    Insert debugs :)
    Do this before and after every linalg generic operation.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        kernel_type = op.body.block.first_op
        assert kernel_type
        kernel_name = kernel_type.name
        kernel_name = kernel_name.replace(".", "_")

        memreftype = op.inputs[0].type
        if not isinstance(memreftype, builtin.MemRefType):
            return

        if isinstance(memreftype.memory_space, builtin.StringAttr):
            level = memreftype.memory_space.data
        else:
            # defaulting to L3
            level = "L3"

        debug_before = debug.DebugLinalgOp(
            op.inputs[0], op.inputs[-1], op.outputs[0], kernel_name, "before", level
        )
        debug_after = debug.DebugLinalgOp(
            op.inputs[0], op.inputs[-1], op.outputs[0], kernel_name, "after", level
        )
        rewriter.insert_op(debug_before, InsertPoint.before(op))
        rewriter.insert_op(debug_after, InsertPoint.after(op))


class InsertDebugPass(ModulePass):
    name = "test-insert-debugs"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            InsertDebugStatements(), apply_recursively=False
        ).rewrite_module(op)
