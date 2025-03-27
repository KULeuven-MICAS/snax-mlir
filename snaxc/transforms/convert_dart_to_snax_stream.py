from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, builtin
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from snaxc.accelerators import AccContext
from snaxc.accelerators.snax import SNAXStreamer
from snaxc.dialects import dart, snax_stream
from snaxc.ir.dart.affine_transform import AffineTransform


@dataclass
class ConvertStreamToSnaxStreamPattern(RewritePattern):
    """
    Convert stream access patterns (with affinemap patterns mapping the iteration
    space to memory) into actual stride pattens for SNAX Streamers, with given
    spatial and temporal strides ready to be programmed through CSRs.
    """

    ctx: AccContext

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dart.AccessPatternOp, rewriter: PatternRewriter):
        assert op.accelerator
        accelerator_type = self.ctx.get_acc(op.accelerator.data)
        assert isinstance(accelerator_type, SNAXStreamer)
        template = accelerator_type.get_template(op)

        snax_stride_patterns: list[snax_stream.StridePattern] = []

        for operand in range(len(op.operands)):
            pattern = AffineTransform.from_affine_map(op.patterns.data[operand].data)

            # Create iterator for all dimensions of the access_mem_map that returns (stride, bound)
            # in reverse, because we work outermost -> innermost and streamers the other way around
            access_iter = iter(
                (int(pattern.A[0, i]), op.bounds.data[i].value.data)
                for i in reversed(range(pattern.num_dims))
            )

            # Fetch the first stride
            stride, bound = next(access_iter)

            temporal_strides: list[int] = []
            spatial_strides: list[int] = []
            upper_bounds: list[int] = []

            # fill up all spatial strides
            for _ in [x for x in template[0].bounds if x is not None]:
                # FIXME: provide more general solution for new spatial streamer_config
                # configuration, this works in all current cases and layouts but is far from generally correct.
                spatial_strides = [8]
                stride, bound = next(access_iter, (None, None))

            # remaining are temporal strides
            while stride is not None and bound is not None:
                temporal_strides.append(stride)
                upper_bounds.append(bound)
                stride, bound = next(access_iter, (None, None))

            # create the stride pattern for this operand
            snax_stride_pattern = snax_stream.StridePattern(
                upper_bounds=upper_bounds,
                temporal_strides=temporal_strides,
                spatial_strides=spatial_strides,
            )
            snax_stride_patterns.append(snax_stride_pattern)

        # TODO: what is still required is a better system for the unused operands
        # of snax_gemmx / other accelerators. this now fills in empty/zero patterns for the unused operands.

        new_inputs: list[SSAValue] = list(op.inputs)
        new_outputs: list[SSAValue] = list(op.outputs)
        ops_to_add: list[Operation] = []

        assert op.accelerator
        if op.accelerator.data == "snax_gemmx":
            empty_pattern = snax_stream.StridePattern(
                upper_bounds=[0] * 3, temporal_strides=[0] * 3, spatial_strides=[0]
            )
            if len(snax_stride_patterns) == 3:
                # matmul

                # for a matmul, the 8-bit output port D8 and the bias in put C
                # are unused, so we create empty patterns for them here.

                # insert empty patterns for D8 and zero pattern for C
                snax_stride_patterns.insert(2, empty_pattern)
                new_inputs.append(op.outputs[0])

                # insert same pattern for C as for D32
                snax_stride_patterns.insert(
                    3,
                    snax_stream.StridePattern(
                        upper_bounds=snax_stride_patterns[3].upper_bounds,
                        temporal_strides=snax_stride_patterns[3].temporal_strides,
                        spatial_strides=[8],
                    ),
                )

                # point C to D32
                new_inputs.append(op.outputs[0])

            elif len(snax_stride_patterns) == 4:
                # gemm
                #
                # for a gemm, the 8bit-output port D8 are unused, so we create
                # empty patterns for them here
                snax_stride_patterns.insert(2, empty_pattern)
                new_inputs.insert(2, op.inputs[-1])

            else:
                # simd
                # to calculate only simd, we calculate the result
                # of D8 = rescale(AxB + C)
                # create zero patterns for A and B such that D8 = rescale(C)
                # create empty pattern for D32
                # do not use new outputs
                new_inputs.append(new_outputs.pop())

                zero_pattern = snax_stream.StridePattern(
                    upper_bounds=snax_stride_patterns[0].upper_bounds,
                    temporal_strides=[0] * len(snax_stride_patterns[0].upper_bounds),
                    spatial_strides=[8],
                )

                # read zeros from tcdm (must make sure there are zeros at these addresses)
                # in the new streamer this can be fixed with byte masking
                snax_stride_patterns.insert(0, zero_pattern)
                ops_to_add.append(
                    ptr := arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
                )
                new_inputs.insert(0, ptr.result)
                snax_stride_patterns.insert(1, zero_pattern)
                ops_to_add.append(
                    ptr := arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
                )
                new_inputs.insert(1, ptr.result)

                # flip D8 and C such that they are in the right order
                snax_stride_patterns.append(snax_stride_patterns.pop(2))
                new_inputs.append(new_inputs.pop(2))

                # empty pattern for D32
                snax_stride_patterns.append(empty_pattern)
                # dummy base pointer for D32
                new_inputs.append(op.inputs[-1])

        if op.accelerator.data == "snax_gemmx":
            # make last spatial stride patterns 2d
            snax_stride_patterns[-2] = snax_stream.StridePattern(
                upper_bounds=snax_stride_patterns[-2].upper_bounds,
                temporal_strides=snax_stride_patterns[-2].temporal_strides,
                spatial_strides=[8, 64],
            )
            snax_stride_patterns[-1] = snax_stream.StridePattern(
                upper_bounds=snax_stride_patterns[-1].upper_bounds,
                temporal_strides=snax_stride_patterns[-1].temporal_strides,
                spatial_strides=[8, 64],
            )

        snax_stride_patterns = [
            pattern.canonicalize() for pattern in snax_stride_patterns
        ]

        # now create snax_streaming region op
        new_op = snax_stream.StreamingRegionOp(
            inputs=new_inputs,
            outputs=new_outputs,
            stride_patterns=snax_stride_patterns,
            accelerator=op.accelerator.data,
            body=rewriter.move_region_contents_to_new_regions(op.body),
        )

        rewriter.replace_matched_op([*ops_to_add, new_op], new_op.results)


@dataclass(frozen=True)
class ConvertDartToSnaxStream(ModulePass):
    name = "convert-dart-to-snax-stream"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        assert isinstance(ctx, AccContext)
        PatternRewriteWalker(ConvertStreamToSnaxStreamPattern(ctx)).rewrite_module(op)
