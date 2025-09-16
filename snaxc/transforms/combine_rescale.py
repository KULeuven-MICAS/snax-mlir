from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, tosa
from xdsl.dialects.builtin import BoolAttr, DenseArrayBase, IntegerAttr, i1, i8, i32
from xdsl.ir import Operation
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

                prev_multiplier = prev_rescale_op.multiplier.get_values()[0]
                prev_shift = prev_rescale_op.shift.get_values()[0]
                current_multiplier = rescale_op.multiplier.get_values()[0]
                current_shift = rescale_op.shift.get_values()[0]
                assert isinstance(prev_multiplier, int)
                assert isinstance(current_multiplier, int)
                assert isinstance(prev_shift, int)
                assert isinstance(current_shift, int)

                new_multiplier = prev_multiplier * current_multiplier
                new_shift = prev_shift + current_shift

                # If the new multiplier is too large, we need to adjust it
                while new_multiplier >= 2**31:
                    new_multiplier //= 2
                    new_shift -= 1

                # Create a new combined rescale operation
                new_rescale_op = tosa.RescaleOp(
                    operands=[new_input],
                    result_types=[new_output_type],
                    properties={
                        "input_zp": IntegerAttr(
                            prev_rescale_op.input_zp.value.data
                            + rescale_op.input_zp.value.data * (2**current_shift) // current_multiplier,
                            i32,
                        ),
                        "output_zp": IntegerAttr(
                            rescale_op.output_zp.value.data
                            + prev_rescale_op.output_zp.value.data * prev_multiplier // (2**prev_shift),
                            i32,
                        ),
                        "multiplier": DenseArrayBase.from_list(
                            i32,
                            [new_multiplier],
                        ),
                        "shift": DenseArrayBase.from_list(
                            i8,
                            [new_shift],
                        ),
                        "scale32": BoolAttr(True, i1),
                        "double_round": BoolAttr(True, i1),
                        "per_channel": BoolAttr(False, i1),
                    },
                )

                # Replace the old ops with the new combined op
                rewriter.replace_op(rescale_op, new_rescale_op)
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
