from __future__ import annotations

from typing import cast

from xdsl.context import Context
from xdsl.dialects import arith
from xdsl.dialects.builtin import (
    AnyFloat,
    BFloat16Type,
    Float16Type,
    Float32Type,
    Float64Type,
    IntegerType,
    ModuleOp,
    SignednessAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Attribute, Operation
from xdsl.irdl import isa
from xdsl.parser import Signedness
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from snaxc.dialects import hardfloat

_type_mapping: dict[type[Attribute], tuple[int, int]] = {
    Float64Type: (11, 53),
    Float32Type: (8, 24),
    Float16Type: (5, 11),
    BFloat16Type: (8, 8),
}

# Op conversion
_arith_to_hardfloat: dict[type[Operation], type[hardfloat.HardfloatOperation]] = {
    arith.AddfOp: hardfloat.AddRecFnOp,
    arith.MulfOp: hardfloat.MulRecFnOp,
}


class ConvertFloatBinaryOps(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.FloatingPointLikeBinaryOperation, rewriter: PatternRewriter):
        # Get sig_width and exp_witdh:
        in_type = op.lhs.type
        if type(in_type) not in _type_mapping:
            return
        exp_width, sig_width = _type_mapping[type(in_type)]
        assert isa(in_type, AnyFloat)
        bitwidth = in_type.bitwidth
        if type(op) not in _arith_to_hardfloat:
            return
        CoreOp = _arith_to_hardfloat[type(op)]

        # Create the recode - core_op - unrecode sandwich
        new_ops = [
            cast_lhs := UnrealizedConversionCastOp.get([op.lhs], [IntegerType(bitwidth)]),
            cast_rhs := UnrealizedConversionCastOp.get([op.rhs], [IntegerType(bitwidth)]),
            recode_lhs := hardfloat.FnToRecFnOp([cast_lhs], [IntegerType(bitwidth + 1)], sig_width, exp_width),
            recode_rhs := hardfloat.FnToRecFnOp([cast_rhs], [IntegerType(bitwidth + 1)], sig_width, exp_width),
            core_op := CoreOp([recode_lhs, recode_rhs], [IntegerType(bitwidth + 1)], sig_width, exp_width),
            unrecode := hardfloat.RecFnToFnOp([core_op], [IntegerType(bitwidth)], sig_width, exp_width),
            cast_res := UnrealizedConversionCastOp.get([unrecode], [in_type]),
        ]
        rewriter.replace_op(op, new_ops=new_ops, new_results=[cast_res.results[0]])


class ConvertIToFPOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.IntegerToFloatingPointBaseOp, rewriter: PatternRewriter):
        match op.name:
            case arith.SIToFPOp.name:
                props = {"signedness": cast(Attribute, SignednessAttr.get(Signedness.SIGNED))}
            case arith.UIToFPOp.name:
                props = {"signedness": cast(Attribute, SignednessAttr.get(Signedness.UNSIGNED))}
            case _:
                raise NotImplementedError()
        exp_width, sig_width = _type_mapping[type(op.result.type)]
        bitwidth = cast(IntegerType, op.input.type).bitwidth
        new_ops = [
            rec_fn := hardfloat.InToRecFnOp(
                [op.input], [IntegerType(bitwidth + 1)], sig_width, exp_width, bitwidth, prop_dict=props
            ),
            unrecode := hardfloat.RecFnToFnOp([rec_fn], [IntegerType(bitwidth)], sig_width, exp_width),
            cast_res := UnrealizedConversionCastOp.get([unrecode], [op.result.type]),
        ]
        rewriter.replace_op(op, new_ops=new_ops, new_results=[cast_res.results[0]])


class ConvertFPToIOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.FloatingPointToIntegerBaseOp, rewriter: PatternRewriter):
        match op.name:
            case arith.FPToSIOp.name:
                props = {"signedness": cast(Attribute, SignednessAttr.get(Signedness.SIGNED))}
            case arith.FPToUIOp.name:
                props = {"signedness": cast(Attribute, SignednessAttr.get(Signedness.UNSIGNED))}
            case _:
                raise NotImplementedError()
        exp_width, sig_width = _type_mapping[type(op.input.type)]
        bitwidth = cast(IntegerType, op.input.type).bitwidth
        new_ops = [
            cast_res := UnrealizedConversionCastOp.get([op.input], [op.result.type]),
            recode := hardfloat.FnToRecFnOp([cast_res], [IntegerType(bitwidth + 1)], sig_width, exp_width),
            rec_fn := hardfloat.RecFnToInOp(
                [recode], [IntegerType(bitwidth)], sig_width, exp_width, bitwidth, prop_dict=props
            ),
        ]
        rewriter.replace_op(op, new_ops=new_ops, new_results=[rec_fn.results[0]])


class ConvertFloatToHardfloatPass(ModulePass):
    name = "convert-float-to-hardfloat"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ConvertFloatBinaryOps(), ConvertIToFPOp(), ConvertFPToIOp()]),
            apply_recursively=False,
        ).rewrite_module(op)
