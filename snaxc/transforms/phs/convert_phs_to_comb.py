from xdsl.context import Context
from xdsl.dialects import arith, builtin, comb
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


MaybeCombBinOp = type[comb.BinCombOperation] | type[comb.VariadicCombOperation]

conversion_table: dict[type[arith.SignlessIntegerBinaryOperation], MaybeCombBinOp] = {
    arith.AddiOp: comb.AddOp,
    arith.MuliOp: comb.MulOp,
    arith.DivUIOp: comb.DivUOp,
    arith.SubiOp: comb.SubOp,
}


class ConvertArithOps(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, bin_op: arith.SignlessIntegerBinaryOperation, rewriter: PatternRewriter):
        new_op_cls = conversion_table[type(bin_op)]
        if issubclass(new_op_cls, comb.BinCombOperation):
            new_op = new_op_cls(operand1=bin_op.lhs, operand2=bin_op.rhs)
        else:  # issubclass(new_op_cls, comb.VariadicCombOperation)
            new_op = new_op_cls(input_list=[bin_op.lhs, bin_op.rhs])
        rewriter.replace_op(bin_op, new_op)


class ConvertPhsToCombPass(ModulePass):
    name = "convert-phs-to-comb"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ConvertMuxes(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(ConvertArithOps(), apply_recursively=False).rewrite_module(op)
