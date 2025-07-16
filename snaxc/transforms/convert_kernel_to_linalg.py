from xdsl.context import Context
from xdsl.dialects import builtin, linalg
from xdsl.dialects.arith import AddiOp, ConstantOp, ExtSIOp, MaxSIOp, MinSIOp, MuliOp, ShRSIOp, SubiOp, TruncIOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from snaxc.dialects.kernel import Parsable, RescaleOp


class LowerRescale(RewritePattern):
    """
    Limited lowering of rescale to linalg,
    ignoring separate channels and double rounding.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: RescaleOp, rewriter: PatternRewriter):
        if not isinstance(linalg_op := op.parent_op(), linalg.GenericOp):
            return

        # create constant ops:
        zp_in = ConstantOp.from_int_and_width(op.input_zp.value.data, builtin.IntegerType(32))
        zp_out = ConstantOp.from_int_and_width(op.output_zp.value.data, builtin.IntegerType(32))
        shift = ConstantOp.from_int_and_width(int(op.shift.get_values()[0]), builtin.IntegerType(64))
        mult = ConstantOp.from_int_and_width(int(op.multiplier.get_values()[0]), builtin.IntegerType(64))
        min = ConstantOp.from_int_and_width(op.min_int.value.data, builtin.IntegerType(32))
        max = ConstantOp.from_int_and_width(op.max_int.value.data, builtin.IntegerType(32))
        rewriter.insert_op([zp_in, zp_out, shift, mult, min, max], InsertPoint.before(linalg_op))

        # create body ops:
        with_zp_in = SubiOp(op.input, zp_in)
        extended = ExtSIOp(with_zp_in, builtin.i64)
        multed = MuliOp(extended, mult)
        shifted = ShRSIOp(multed, shift)
        trunced = TruncIOp(shifted, builtin.i32)
        with_zp_out = AddiOp(trunced, zp_out)
        clamped_max = MinSIOp(with_zp_out, max)
        clamped_min = MaxSIOp(clamped_max, min)
        trunced_final = TruncIOp(clamped_min, builtin.i8)
        rewriter.replace_matched_op(
            [with_zp_in, extended, multed, shifted, trunced, with_zp_out, clamped_max, clamped_min, trunced_final]
        )


class LowerLinalgBody(RewritePattern):
    """
    Matches on linalg.generic operations to check if
    their body is kernel op defined in the kernel dialect.
    Replaces the body with the equivalent arith body if this is true.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, linalg_op: linalg.GenericOp, rewriter: PatternRewriter):
        # find the kernel op in linalg body
        if not isinstance(kernel_op := linalg_op.body.block.first_op, Parsable):
            return

        # only works for non-fused kernels (only 1 kernel op)
        if not isinstance(kernel_op.next_op, linalg.YieldOp):
            return

        # replace linalg op
        rewriter.replace_matched_op(
            linalg.GenericOp(
                linalg_op.inputs,
                linalg_op.outputs,
                kernel_op.equivalent_region,
                linalg_op.indexing_maps,
                linalg_op.iterator_types,
                linalg_op.result_types,
                linalg_op.library_call,
                linalg_op.doc,
            )
        )


class ConvertKernelToLinalg(ModulePass):
    name = "convert-kernel-to-linalg"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(LowerLinalgBody()).rewrite_module(op)
        PatternRewriteWalker(LowerRescale()).rewrite_module(op)
