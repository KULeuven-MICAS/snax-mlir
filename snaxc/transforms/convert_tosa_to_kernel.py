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
        if not isinstance(
            clamp_op := next(iter(rescale_op.output.uses)).operation, tosa.ClampOp
        ):
            # no clamping op after, so we integrate clamping in rescale op to int8 range
            # iff the output of the rescale op is int8
            if (
                rescale_op.output.type.element_type != builtin.i8
                and rescale_op.output.type.element_type != builtin.i32
            ):
                return
            clamp_op = rescale_op

        # should have tensor inputs
        if not isa(inp_type := rescale_op.input.type, builtin.TensorType[Attribute]):
            return
        if not isa(out_type := clamp_op.output.type, builtin.TensorType[Attribute]):
            return

        # create linalg body with kernel op with the params of tosa ops

        # Extract all values:
        builtin.i32.verify_value(rescale_op.input_zp.value.data)
        builtin.i32.verify_value(rescale_op.output_zp.value.data)
        input_zp = rescale_op.input_zp.value.data
        output_zp = rescale_op.output_zp.value.data
        multiplier = rescale_op.multiplier.get_values()
        assert isa(multiplier, tuple[int, ...])
        shift = rescale_op.shift.get_values()
        assert isa(shift, tuple[int, ...])
        if isinstance(clamp_op, tosa.ClampOp):
            if rescale_op.output.type.element_type == builtin.i8:
                builtin.i8.verify_value(clamp_op.max_int.value.data)
                builtin.i8.verify_value(clamp_op.min_int.value.data)
                max_int = clamp_op.max_int.value.data
                min_int = clamp_op.min_int.value.data
            else:
                builtin.i32.verify_value(clamp_op.max_int.value.data)
                builtin.i32.verify_value(clamp_op.min_int.value.data)
                max_int = clamp_op.max_int.value.data
                min_int = clamp_op.min_int.value.data
        else:
            assert isinstance(rescale_op.output.type.element_type, builtin.IntegerType)
            min_int, max_int = rescale_op.output.type.element_type.value_range()
            max_int = max_int // 2 - 1
        double_round = rescale_op.double_round.value.data
        assert double_round in (0, -1)

        @Builder.implicit_region((inp_type.element_type, out_type.element_type))
        def linalg_body(args: tuple[BlockArgument, ...]) -> None:
            kernel_op = kernel.RescaleOp(
                args[0],
                args[-1].type,
                input_zp,
                output_zp,
                multiplier,
                shift,
                max_int,
                min_int,
                bool(double_round),
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
                dim_idx_ops.append(
                    dim_idx := arith.ConstantOp.from_int_and_width(
                        dim_idx, builtin.IndexType()
                    )
                )
                dim_ops.append(tensor.DimOp(rescale_op.input, dim_idx))

        dim_op_values = [dim_op.result for dim_op in dim_ops]
        output_tensor = tensor.EmptyOp(dim_op_values, out_type)

        new_op = linalg.GenericOp(
            inputs=[rescale_op.input],
            outputs=[output_tensor.tensor],
            body=linalg_body,
            indexing_maps=[
                builtin.AffineMapAttr(
                    AffineMap(
                        nb_dims, 0, tuple(AffineDimExpr(i) for i in range(nb_dims))
                    )
                )
                for _ in range(2)
            ],
            iterator_types=builtin.ArrayAttr(
                [linalg.IteratorTypeAttr.parallel()] * nb_dims
            ),
            result_types=(clamp_op.output.type,),
        )

        # insert new op
        rewriter.replace_op(clamp_op, (*dim_idx_ops, *dim_ops, output_tensor, new_op))
        if rescale_op is not clamp_op:
            rewriter.erase_matched_op()


class AvgPoolPattern(RewritePattern):
    """
    Transform tosa AvgPool2DOp into kernel.AvgPool2DOp
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, avgpool_op: tosa.AvgPool2DOp, rewriter: PatternRewriter
    ):
        # should have tensor inputs
        if not isa(inp_type := avgpool_op.input.type, builtin.TensorType[Attribute]):
            return

        # Extract all values:
        kernel_tuple = avgpool_op.kernel.get_values()
        assert isa(kernel_tuple, tuple[int, ...])
        stride = avgpool_op.stride.get_values()
        assert isa(stride, tuple[int, ...])
        pad = avgpool_op.pad.get_values()
        assert isa(pad, tuple[int, ...])

        kernel_size = kernel_tuple[0] * kernel_tuple[1]

        # create linalg body with kernel op with the params of tosa ops
        @Builder.implicit_region((inp_type.element_type, inp_type.element_type, inp_type.element_type))
        def linalg_body(args: tuple[BlockArgument, ...]) -> None:
            kernel_op = kernel.AvgPoolOp(
                args[0],
                args[-1],
                args[-1].type,
                kernel_size,
            )
            linalg.YieldOp(kernel_op)

        # create elementwise linalg op
        nb_dims = inp_type.get_num_dims()
        assert nb_dims == 4, "AvgPool2DOp expects a 4D tensor input"

        # destination type:
        dim_idx_ops: list[arith.ConstantOp] = []
        dim_ops: list[tensor.DimOp] = []
        for dim_idx, shape in enumerate(inp_type.get_shape()):
            if shape == -1:
                # create dim op
                dim_idx_ops.append(
                    dim_idx := arith.ConstantOp.from_int_and_width(
                        dim_idx, builtin.IndexType()
                    )
                )
                dim_ops.append(tensor.DimOp(avgpool_op.input, dim_idx))

        dim_op_values = [dim_op.result for dim_op in dim_ops]

        outp_shape = inp_type.get_shape()
        batch, m, n, channels = (
            outp_shape[0],
            outp_shape[1],
            outp_shape[2],
            outp_shape[3],
        )
        assert channels % 64 == 0, "Channels must be a multiple of 64"
        output_m = (m - kernel_tuple[0] + pad[2]) // stride[0] + 1
        output_n = (n - kernel_tuple[1] + pad[3]) // stride[1] + 1
        output_shape = (batch, output_m, output_n, channels)
        output_type = builtin.TensorType(inp_type.element_type, output_shape)
        output_tensor = tensor.EmptyOp(dim_op_values, output_type)

        kernel_tensor_type = builtin.TensorType(inp_type.element_type, kernel_tuple)
        kernel_tensor_op = tensor.EmptyOp([], kernel_tensor_type)

        new_op = linalg.GenericOp(
            inputs=[avgpool_op.input, kernel_tensor_op.results[0]],
            outputs=[output_tensor.tensor],
            body=linalg_body,
            indexing_maps=[
                # Input tensor map: (d0, ((d1 * 2) + d4), ((d2 * 2) + d5), d3)
                builtin.AffineMapAttr(
                    AffineMap(
                        6,
                        0,
                        (
                            AffineDimExpr(0),  # d0
                            AffineDimExpr(1) * 2 + AffineDimExpr(4),  # ((d1 * 2) + d4)
                            AffineDimExpr(2) * 2 + AffineDimExpr(5),  # ((d2 * 2) + d5)
                            AffineDimExpr(3),  # d3
                        ),
                    )
                ),
                # Kernel tensor map: (d4, d5)
                builtin.AffineMapAttr(
                    AffineMap(
                        6,
                        0,
                        (
                            AffineDimExpr(4),  # d4
                            AffineDimExpr(5),  # d5
                        ),
                    )
                ),
                # Output tensor map: (d0, d1, d2, d3)
                builtin.AffineMapAttr(
                    AffineMap(
                        6,
                        0,
                        (
                            AffineDimExpr(0),  # d0
                            AffineDimExpr(1),  # d1
                            AffineDimExpr(2),  # d2
                            AffineDimExpr(3),  # d3
                        ),
                    )
                ),
            ],
            iterator_types=builtin.ArrayAttr(
                [linalg.IteratorTypeAttr.parallel()] * 4
                + [linalg.IteratorTypeAttr.reduction()] * 2
            ),
            result_types=(avgpool_op.output.type,),
        )

        # insert new op
        rewriter.replace_op(
            avgpool_op,
            (*dim_idx_ops, *dim_ops, output_tensor, kernel_tensor_op, new_op),
        )


class ConvertTosaToKernelPass(ModulePass):
    """
    Converts tosa dialect ops to kernel ops wrapped in linalg generics.
    """

    name = "convert-tosa-to-kernel"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(RescaleClampPattern()).rewrite_module(op)
        PatternRewriteWalker(AvgPoolPattern()).rewrite_module(op)
