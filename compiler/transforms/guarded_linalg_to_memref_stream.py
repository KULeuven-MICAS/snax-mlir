from xdsl.context import MLContext
from xdsl.dialects import linalg, memref_stream
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.convert_linalg_to_memref_stream import (
    ConvertGenericOpPattern,
    ConvertYieldOpPattern,
)


class GuardedGenericOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter) -> None:
        """
        This pattern is a guarded version of the pattern in upstream xDSL.
        This is required because we can't (or don't want to) lower all
        generic calls through the stream dialects.
        It only matches on linalg.generics that have a library call with
        a `_stream` suffix. This pass removes that suffix.
        """

        if not op.library_call:
            return

        if op.library_call.data.endswith("_stream"):
            op.library_call = StringAttr(op.library_call.data[: -len("_stream")])
            ConvertGenericOpPattern().match_and_rewrite(op, rewriter)

class GuardedYieldOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.YieldOp, rewriter: PatternRewriter):
        # find parent operation
        parent_op = op.parent
        while parent_op and not isinstance(parent_op, Operation):
            parent_op = parent_op.parent

        # only if parrent is memref generic
        if isinstance(parent_op, memref_stream.GenericOp):
            ConvertYieldOpPattern().match_and_rewrite(op, rewriter)


class GuardedLinalgToMemrefStreamPass(ModulePass):
    name = "guarded-linalg-to-memref-stream"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [GuardedGenericOpPattern(), GuardedYieldOpPattern()]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
