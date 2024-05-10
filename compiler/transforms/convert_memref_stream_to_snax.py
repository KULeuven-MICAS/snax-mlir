from xdsl.dialects import (
    memref_stream,
)
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class StreamOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.StreamingRegionOp, rewriter: PatternRewriter
    ):
        breakpoint()


class ConvertMemrefStreamToSnax(ModulePass):
    name = "convert-memref-stream-to-snax"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    StreamOpLowering(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
