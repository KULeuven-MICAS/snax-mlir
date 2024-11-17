from xdsl.context import MLContext
from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import i8
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.dialects import kernel


class RescaleToTrunc(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: kernel.RescaleOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(arith.TruncIOp(op.input, i8))


class TestRescaleToTrunc(ModulePass):
    name = "test-rescale-to-trunc"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(RescaleToTrunc()).rewrite_module(op)
