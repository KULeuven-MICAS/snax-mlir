from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.parser import AnyFloat
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa

from snaxc.dialects import phs


class CastDataOperands(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, pe: phs.PEOp, rewriter: PatternRewriter):
        for data_opnd in pe.data_operands():
            typ = data_opnd.type
            if isa(typ, AnyFloat):
                data_opnd = rewriter.replace_value_with_new_type(data_opnd, builtin.IntegerType(typ.bitwidth))
        pe.function_type = builtin.FunctionType.from_lists(
            [*(op.type for op in pe.data_operands()), *(builtin.IndexType() for _ in range(pe.switch_no.value.data))],
            [builtin.IntegerType(t.bitwidth) for t in pe.function_type.outputs],
        )


class CastChooseOps(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: phs.ChooseOp, rewriter: PatternRewriter):
        for region in op.regions:
            ip = InsertPoint.at_start(region.block)
            for block_arg in region.block.args:
                old_type = block_arg.type
                if not isa(old_type, AnyFloat):
                    continue
                new_type = builtin.IntegerType(old_type.bitwidth)
                block_arg = rewriter.replace_value_with_new_type(block_arg, new_type)
                cast, new_val = builtin.UnrealizedConversionCastOp.cast_one(block_arg, old_type)
                rewriter.insert_op(cast, ip)
                block_arg.replace_by_if(new_val, lambda u: u.operation is not cast)
        for res in op.results:
            if isa(res.type, AnyFloat):
                rewriter.replace_value_with_new_type(res, builtin.IntegerType(res.type.bitwidth))


class CastYieldOps(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, yieldop: phs.YieldOp, rewriter: PatternRewriter, /):
        for op in yieldop.operands:
            if not isa(op.type, AnyFloat):
                continue
            cast, newr = builtin.UnrealizedConversionCastOp.cast_one(op, builtin.IntegerType(op.type.bitwidth))
            rewriter.insert_op(cast)
            op.replace_by_if(newr, lambda u: u.operation is yieldop)


class PhsConvertFloatToInt(ModulePass):
    name = "phs-convert-float-to-int"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(CastDataOperands(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(CastChooseOps(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(CastYieldOps(), apply_recursively=False).rewrite_module(op)
