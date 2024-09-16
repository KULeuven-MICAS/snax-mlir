from xdsl.builder import Builder
from xdsl.context import MLContext
from xdsl.dialects import builtin, linalg, tosa
from xdsl.dialects.builtin import IntegerAttr, i1, i8, i32
from xdsl.ir import BlockArgument
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.dialects import kernel


class RescaleClampPattern(RewritePattern):
    """
    Transform rescale clamp into a kernel.rescale op
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, rescale_op: tosa.RescaleOp, rewriter: PatternRewriter):
        # searching for the pattern rescale + clamp
        if len(rescale_op.output.uses) != 1:
            return
        if not isinstance(
            clamp_op := next(iter(rescale_op.output.uses)).operation, tosa.ClampOp
        ):
            return

        # should have tensor inputs
        if not isinstance(inp_type := rescale_op.input.type, builtin.TensorType):
            return
        if not isinstance(out_type := clamp_op.output.type, builtin.TensorType):
            return

        # create linalg body with kernel op with the params of tosa ops

        # Extract all values:
        input_zp = rescale_op.input_zp.value.data
        assert -128 <= input_zp
        assert input_zp <= 127

        output_zp = rescale_op.output_zp.value.data
        assert -128 <= output_zp
        assert output_zp <= 127

        multiplier = rescale_op.multiplier.data.data[0].data
        assert isinstance(multiplier, int)

        shift = rescale_op.shift.data.data[0].data
        assert isinstance(shift, int)
        assert -128 <= shift
        assert shift <= 127

        max_int = clamp_op.max_int.value.data
        assert -128 <= max_int
        assert max_int <= 127

        min_int = clamp_op.min_int.value.data
        assert -128 <= min_int
        assert min_int <= 127

        double_round = rescale_op.double_round.value.data
        assert double_round in (0, 1)

        @Builder.implicit_region((inp_type.element_type, out_type.element_type))
        def linalg_body(args: tuple[BlockArgument, ...]) -> None:
            kernel_op = kernel.RescaleOp(
                operands=[args[0]],
                result_types=[args[-1].type],
                properties={
                    "input_zp": IntegerAttr(input_zp, i8),
                    "output_zp": IntegerAttr(output_zp, i8),
                    "multiplier": IntegerAttr(multiplier, i32),
                    "shift": IntegerAttr(shift, i8),
                    "max_int": IntegerAttr(max_int, i8),
                    "min_int": IntegerAttr(min_int, i8),
                    "double_round": IntegerAttr(double_round, i1),
                },
            )
            linalg.YieldOp(kernel_op)

        # create elementwise linalg op
        nb_dims = inp_type.get_num_dims()

        new_op = linalg.Generic(
            inputs=[rescale_op.input],
            outputs=[rescale_op.input],
            body=linalg_body,
            indexing_maps=[
                builtin.AffineMapAttr(
                    AffineMap(
                        nb_dims, 0, tuple(AffineDimExpr(i) for i in range(nb_dims))
                    )
                )
                for _ in range(2)
            ],
            iterator_types=builtin.ArrayAttr([linalg.IteratorTypeAttr.parallel()] * 2),
            result_types=(clamp_op.output.type,),
        )

        # insert new op
        rewriter.replace_op(clamp_op, new_op)
        rewriter.erase_matched_op()


class ConvertTosaToKernelPass(ModulePass):
    """
    Converts tosa dialect ops to kernel ops wrapped in linalg generics.
    """

    name = "convert-tosa-to-kernel"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(RescaleClampPattern()).rewrite_module(op)
