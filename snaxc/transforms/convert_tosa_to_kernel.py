from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, linalg, tensor, tosa
from xdsl.ir import Attribute, BlockArgument
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa

from snaxc.dialects import kernel


def assert_int8(val: float | int) -> int:
    assert isinstance(val, int)
    assert isinstance(val, int)
    assert -128 <= val
    assert val <= 127
    return val


class RescaleClampPattern(RewritePattern):
    """
    Transform rescale clamp into a kernel.rescale op
    If there is no clamp op after the rescaling op,
    the clamping defaults to the range of the int8 output type (-128, 127)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, rescale_op: tosa.RescaleOp, rewriter: PatternRewriter):
        # searching for the pattern rescale + clamp
        if len(rescale_op.output.uses) != 1:
            return
        if not isinstance(clamp_op := next(iter(rescale_op.output.uses)).operation, tosa.ClampOp):
            # no clamping op after, so we integrate clamping in rescale op to int8 range
            clamp_op = rescale_op

        # should have tensor inputs
        if not isa(inp_type := rescale_op.input.type, builtin.TensorType[Attribute]):
            return
        if not isa(out_type := clamp_op.output.type, builtin.TensorType[Attribute]):
            return

        # create linalg body with kernel op with the params of tosa ops

        # Extract all values:
        input_zp = assert_int8(rescale_op.input_zp.value.data)
        output_zp = assert_int8(rescale_op.output_zp.value.data)
        multiplier = rescale_op.multiplier.get_values()[0]
        assert isinstance(multiplier, int)
        shift = assert_int8(rescale_op.shift.get_values()[0])
        if isinstance(clamp_op, tosa.ClampOp):
            max_int = assert_int8(clamp_op.max_int.value.data)
            min_int = assert_int8(clamp_op.min_int.value.data)
        else:
            max_int = 127
            min_int = -128
        double_round = rescale_op.double_round.value.data
        assert double_round in (0, -1)

        @Builder.implicit_region((inp_type.element_type, out_type.element_type))
        def linalg_body(args: tuple[BlockArgument, ...]) -> None:
            kernel_op = kernel.RescaleOp(
                args[0], args[-1].type, input_zp, output_zp, multiplier, shift, max_int, min_int, bool(double_round)
            )
            linalg.YieldOp(kernel_op)

        # create elementwise linalg op
        nb_dims = inp_type.get_num_dims()

        # destination type:
        dim_idx_ops: list[arith.ConstantOp] = []
        dim_ops: list[tensor.DimOp] = []
        for dim_idx, shape in enumerate(out_type.get_shape()):
            if shape == -1:
                # create dim op
                dim_idx_ops.append(dim_idx := arith.ConstantOp.from_int_and_width(dim_idx, builtin.IndexType()))
                dim_ops.append(tensor.DimOp(rescale_op.input, dim_idx))

        dim_op_values = [dim_op.result for dim_op in dim_ops]
        output_tensor = tensor.EmptyOp(dim_op_values, out_type)

        new_op = linalg.GenericOp(
            inputs=[rescale_op.input],
            outputs=[output_tensor.tensor],
            body=linalg_body,
            indexing_maps=[
                builtin.AffineMapAttr(AffineMap(nb_dims, 0, tuple(AffineDimExpr(i) for i in range(nb_dims))))
                for _ in range(2)
            ],
            iterator_types=builtin.ArrayAttr([linalg.IteratorTypeAttr.parallel()] * 2),
            result_types=(clamp_op.output.type,),
        )

        # insert new op
        rewriter.replace_op(clamp_op, (*dim_idx_ops, *dim_ops, output_tensor, new_op))
        if rescale_op is not clamp_op:
            rewriter.erase_matched_op()


class ConvertTosaToKernelPass(ModulePass):
    """
    Converts tosa dialect ops to kernel ops wrapped in linalg generics.
    """

    name = "convert-tosa-to-kernel"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(RescaleClampPattern()).rewrite_module(op)
