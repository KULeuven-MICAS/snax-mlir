from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, tosa
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType, i8, i32
from xdsl.ir import Operation, OpResult
from xdsl.irdl import Operand
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class TosaCombineRescaleRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, rescale_op: tosa.RescaleOp, rewriter: PatternRewriter):
        """Combine consecutive rescale operations into a single rescale operation."""

        if isinstance(rescale_op.input.owner, Operation):
            if isinstance(rescale_op.input.owner, tosa.RescaleOp):
                # Only combine if the input does not get used anywhere else
                if rescale_op.input.uses.get_length() != 1:
                    return
                prev_rescale_op = rescale_op.input.owner
                new_input = prev_rescale_op.input
                new_output_type = rescale_op.result_types[0]

                def extract_const(operand: Operand):
                    assert isinstance(operand, OpResult)
                    assert isinstance(const := operand.op, tosa.ConstOp)
                    return const.values.get_values()

                prev_input_zp = extract_const(prev_rescale_op.input_zp)[0]
                prev_output_zp = extract_const(prev_rescale_op.output_zp)[0]
                prev_multiplier = extract_const(prev_rescale_op.multiplier)[0]
                prev_shift = extract_const(prev_rescale_op.shift)[0]
                current_input_zp = extract_const(rescale_op.input_zp)[0]
                current_output_zp = extract_const(rescale_op.output_zp)[0]
                current_multiplier = extract_const(rescale_op.multiplier)[0]
                current_shift = extract_const(rescale_op.shift)[0]
                assert isinstance(prev_input_zp, int)
                assert isinstance(prev_output_zp, int)
                assert isinstance(prev_multiplier, int)
                assert isinstance(prev_shift, int)
                assert isinstance(current_input_zp, int)
                assert isinstance(current_output_zp, int)
                assert isinstance(current_multiplier, int)
                assert isinstance(current_shift, int)

                new_input_zp: int = prev_input_zp + current_input_zp * (2**current_shift) // current_multiplier
                new_output_zp: int = current_output_zp + prev_output_zp * prev_multiplier // (2**prev_shift)
                new_multiplier = prev_multiplier * current_multiplier
                new_shift = prev_shift + current_shift

                # If the new multiplier is too large, we need to adjust it
                while new_multiplier >= 2**31:
                    new_multiplier //= 2
                    new_shift -= 1

                # Create a new combined rescale operation
                input_zp = tosa.ConstOp(DenseIntOrFPElementsAttr.from_list(TensorType(i32, (1,)), (new_input_zp,)))
                output_zp = tosa.ConstOp(DenseIntOrFPElementsAttr.from_list(TensorType(i32, (1,)), (new_output_zp,)))
                multiplier = tosa.ConstOp(DenseIntOrFPElementsAttr.from_list(TensorType(i32, (1,)), (new_multiplier,)))
                shift = tosa.ConstOp(DenseIntOrFPElementsAttr.from_list(TensorType(i8, (1,)), (new_shift,)))
                new_rescale_op = tosa.RescaleOp(
                    operands=[new_input, multiplier, shift, input_zp, output_zp],
                    result_types=[new_output_type],
                    properties={val: attr for (val, attr) in rescale_op.properties.items()},
                )

                # Replace the old ops with the new combined op
                rewriter.replace_op(rescale_op, [input_zp, output_zp, multiplier, shift, new_rescale_op])
                rewriter.erase_op(prev_rescale_op)


@dataclass(frozen=True)
class TosaCombineRescale(ModulePass):
    name = "tosa-combine-rescale"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            TosaCombineRescaleRewriter(),
            apply_recursively=False,
            # First elevate outside first for-loop, then move outside the second for-loop (and optionally more)
        ).rewrite_module(op)
