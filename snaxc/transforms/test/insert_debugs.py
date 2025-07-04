from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, linalg
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from snaxc.dialects import dart
from snaxc.dialects.test import debug


@dataclass(frozen=True)
class InsertDebugStatements(RewritePattern):
    """
    Insert debugs :)
    Do this before and after every linalg generic operation.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
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

        # Note, the debug op expects 2 inputs. If only 1 is available,
        # this just sends the first input twice.
        input1 = op.inputs[0]
        input2 = op.inputs[1] if len(op.inputs) >= 2 else op.inputs[0]

        debug_before = debug.DebugLinalgOp(input1, input2, op.outputs[0], kernel_name, "before", level)
        debug_after = debug.DebugLinalgOp(input1, input2, op.outputs[0], kernel_name, "after", level)
        rewriter.insert_op(debug_before, InsertPoint.before(op))
        rewriter.insert_op(debug_after, InsertPoint.after(op))


@dataclass(frozen=True)
class InsertDebugStatementsDart(RewritePattern):
    """
    Insert debugs for dart ops
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dart.StreamingRegionOpBase, rewriter: PatternRewriter):
        kernel_name = "dart"

        memreftype = op.inputs[0].type
        if not isinstance(memreftype, builtin.MemRefType):
            return

        if isinstance(memreftype.memory_space, builtin.StringAttr):
            level = memreftype.memory_space.data
        else:
            # defaulting to L3
            level = "L3"

        # Note, the debug op expects 2 inputs. If only 1 is available,
        # this just sends the first input twice.
        input1 = op.inputs[0]
        input2 = op.inputs[1] if len(op.inputs) >= 2 else op.inputs[0]

        debug_before = debug.DebugLinalgOp(input1, input2, op.outputs[0], kernel_name, "before", level)
        debug_after = debug.DebugLinalgOp(input1, input2, op.outputs[0], kernel_name, "after", level)
        rewriter.insert_op(debug_before, InsertPoint.before(op))
        rewriter.insert_op(debug_after, InsertPoint.after(op))


class InsertDebugPass(ModulePass):
    name = "test-insert-debugs"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(InsertDebugStatements(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(InsertDebugStatementsDart(), apply_recursively=False).rewrite_module(op)
