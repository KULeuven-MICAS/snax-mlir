from xdsl.context import Context
from xdsl.dialects import builtin, comb
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from snaxc.dialects import phs


class ConvertMuxes(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, mux: phs.MuxOp, rewriter: PatternRewriter):
        # 0 = lhs, 1 = rhs
        casted_switch, _ = builtin.UnrealizedConversionCastOp.cast_one(mux.switch, builtin.IntegerType(1))
        new_mux = comb.MuxOp(casted_switch, mux.rhs, mux.lhs)
        rewriter.replace_op(mux, [casted_switch, new_mux])


class ConvertPhsToCombPass(ModulePass):
    name = "convert-phs-to-comb"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ConvertMuxes(), apply_recursively=False).rewrite_module(op)
