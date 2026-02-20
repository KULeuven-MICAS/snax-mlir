from __future__ import annotations

from typing import cast

from xdsl.context import Context
from xdsl.dialects import arith, hw
from xdsl.dialects.builtin import (
    AnyFloat,
    BFloat16Type,
    Float16Type,
    Float32Type,
    Float64Type,
    IntegerType,
    ModuleOp,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Attribute
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from snaxc.dialects.hardfloat import AddRecFnOp, FnToRecFnOp, InToRecFnOp, MulRecFnOp, RecFnToFnOp, RecFnToInOp

_type_mapping: dict[type[Attribute], tuple[int, int]] = {
    Float64Type: (11, 53),
    Float32Type: (8, 24),
    Float16Type: (5, 11),
    BFloat16Type: (8, 8),
}


class ConvertAddSubOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.AddfOp | arith.SubfOp, rewriter: PatternRewriter):
        # Get sig_width and exp_witdh:
        in_type = cast(AnyFloat, op.lhs.type)  # These are verified by IRDL
        if type(in_type) not in _type_mapping:
            return
        exp_width, sig_width = _type_mapping[type(in_type)]
        bitwidth = in_type.bitwidth
        match op:
            case arith.AddfOp():
                subOp = hw.ConstantOp(0, 1)
            case arith.SubfOp():
                subOp = hw.ConstantOp(1, 1)
        # Create the recode - core_op - unrecode sandwich
        new_ops = [
            subOp,
            cast_lhs := UnrealizedConversionCastOp.get([op.lhs], [IntegerType(bitwidth)]),
            cast_rhs := UnrealizedConversionCastOp.get([op.rhs], [IntegerType(bitwidth)]),
            recode_lhs := FnToRecFnOp([cast_lhs], [IntegerType(bitwidth + 1)], sig_width, exp_width),
            recode_rhs := FnToRecFnOp([cast_rhs], [IntegerType(bitwidth + 1)], sig_width, exp_width),
            rounding_mode := hw.ConstantOp(0, 3),
            tininess := hw.ConstantOp(1, 1),
            add := AddRecFnOp(
                [subOp, recode_lhs, recode_rhs, rounding_mode, tininess],
                [IntegerType(bitwidth + 1), IntegerType(5)],
                sig_width,
                exp_width,
            ),
            unrecode := RecFnToFnOp([add.results[0]], [IntegerType(bitwidth)], sig_width, exp_width),
            cast_res := UnrealizedConversionCastOp.get([unrecode], [in_type]),
        ]
        rewriter.replace_op(op, new_ops=new_ops, new_results=[cast_res.results[0]])


class ConvertMulOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MulfOp, rewriter: PatternRewriter):
        # Get sig_width and exp_witdh:
        in_type = cast(AnyFloat, op.lhs.type)  # These are verified by IRDL
        if type(in_type) not in _type_mapping:
            return
        exp_width, sig_width = _type_mapping[type(in_type)]
        bitwidth = in_type.bitwidth

        # Create the recode - core_op - unrecode sandwich
        new_ops = [
            cast_lhs := UnrealizedConversionCastOp.get([op.lhs], [IntegerType(bitwidth)]),
            cast_rhs := UnrealizedConversionCastOp.get([op.rhs], [IntegerType(bitwidth)]),
            recode_lhs := FnToRecFnOp([cast_lhs], [IntegerType(bitwidth + 1)], sig_width, exp_width),
            recode_rhs := FnToRecFnOp([cast_rhs], [IntegerType(bitwidth + 1)], sig_width, exp_width),
            rounding_mode := hw.ConstantOp(0, 3),
            tininess := hw.ConstantOp(1, 1),
            mul := MulRecFnOp(
                [recode_lhs, recode_rhs, rounding_mode, tininess],
                [IntegerType(bitwidth + 1), IntegerType(5)],
                sig_width,
                exp_width,
            ),
            unrecode := RecFnToFnOp([mul.results[0]], [IntegerType(bitwidth)], sig_width, exp_width),
            cast_res := UnrealizedConversionCastOp.get([unrecode], [in_type]),
        ]
        rewriter.replace_op(op, new_ops=new_ops, new_results=[cast_res.results[0]])


class ConvertIToFPOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SIToFPOp | arith.UIToFPOp, rewriter: PatternRewriter):
        match op:
            case arith.SIToFPOp():
                signed_in = hw.ConstantOp(1, 1)
            case arith.UIToFPOp():
                signed_in = hw.ConstantOp(0, 1)
        exp_width, sig_width = _type_mapping[type(op.result.type)]
        bitwidth = cast(IntegerType, op.input.type).bitwidth
        new_ops = [
            signed_in,
            rounding_mode := hw.ConstantOp(0, 3),
            tininess := hw.ConstantOp(1, 1),
            conversion := InToRecFnOp(
                [signed_in.result, op.input, rounding_mode, tininess],
                [IntegerType(bitwidth + 1), IntegerType(5)],
                sig_width,
                exp_width,
                bitwidth,
            ),
            unrecode := RecFnToFnOp([conversion.results[0]], [IntegerType(bitwidth)], sig_width, exp_width),
            cast_res := UnrealizedConversionCastOp.get([unrecode], [op.result.type]),
        ]
        rewriter.replace_op(op, new_ops=new_ops, new_results=[cast_res.results[0]])


class ConvertFPToIOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.FloatingPointToIntegerBaseOp, rewriter: PatternRewriter):
        match op.name:
            case arith.FPToSIOp.name:
                signed_out = hw.ConstantOp(1, 1)
            case arith.FPToUIOp.name:
                signed_out = hw.ConstantOp(0, 1)
            case _:
                raise NotImplementedError()
        exp_width, sig_width = _type_mapping[type(op.input.type)]
        bitwidth = cast(IntegerType, op.input.type).bitwidth
        new_ops = [
            signed_out,
            rounding_mode := hw.ConstantOp(0, 3),
            cast_res := UnrealizedConversionCastOp.get([op.input], [op.result.type]),
            recode := FnToRecFnOp([cast_res], [IntegerType(bitwidth + 1)], sig_width, exp_width),
            rec_fn := RecFnToInOp(
                [recode, rounding_mode, signed_out],
                [IntegerType(bitwidth), IntegerType(3)],
                sig_width,
                exp_width,
                bitwidth,
            ),
        ]
        rewriter.replace_op(op, new_ops=new_ops, new_results=[rec_fn.results[0]])


class ConvertFloatToHardfloatPass(ModulePass):
    name = "convert-float-to-hardfloat"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ConvertAddSubOp(), ConvertMulOp(), ConvertIToFPOp(), ConvertFPToIOp()]),
            apply_recursively=False,
        ).rewrite_module(op)
