from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from snaxc.accelerators import AccContext
from snaxc.accelerators.snax import SNAXStreamer
from snaxc.accelerators.streamers.streamers import StreamerOpts
from snaxc.dialects import dart, snax_stream
from snaxc.ir.dart.affine_transform import AffineTransform

TCDM_BANK_WIDTH = 8


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
        streamers = accelerator_type.get_streamers(op)

        snax_stride_patterns: list[snax_stream.StridePattern] = []

        for operand in range(len(op.operands)):
            pattern = AffineTransform.from_affine_map(op.patterns.data[operand].data)

            # Create iterator for all dimensions of the access_mem_map that returns (stride, bound)
            # in reverse, because we work outermost -> innermost and streamers the other way around

            # do not generate stride, bound pairs for irrelevant spatial dimensions
            # all temporal dimensions are relevant for access patterns:
            relevant: list[bool] = [True] * (pattern.num_dims - template.num_dims)
            # relevant spatial strides have a component in the template matrix
            relevant += template[operand].pattern.A.any(axis=0).tolist()

            access_iter = iter(
                (int(pattern.A[0, i]), op.bounds.data[i].value.data)
                for i in reversed(range(pattern.num_dims))
                if relevant[i]
            )

            temporal_strides: list[int] = []
            spatial_strides: list[int] = []
            upper_bounds: list[int] = []

            # Fetch the first stride
            stride, bound = next(access_iter)

            # TCDM takes 8 contiguous bytes minimum
            if stride * bound == TCDM_BANK_WIDTH:
                stride, bound = next(access_iter)
            else:
                stride, bound = TCDM_BANK_WIDTH, (stride * bound) // TCDM_BANK_WIDTH

            # fill up all spatial strides
            for spat_size in streamers[operand].spatial_dims:
                spatial_strides.append(stride)
                if bound == spat_size:
                    # nice, strides correspond with streamer dimension
                    stride, bound = next(access_iter)
                elif bound < spat_size:
                    # caution! the dimensions of the stride pattern don't nicely overlap with the dimensions
                    # of the streamers, this is only allowed if the two pattern dimensions can be merged.
                    # here, we try to let the streamer take the stride (but it will fetch too many elements).
                    # this will result in an applied_stride and applied_bound that overlaps with the next
                    # stride pattern dimension, it should be checked that this is still correct.
                    assert spat_size % bound == 0
                    applied_stride = stride * bound
                    applied_bound = spat_size // bound
                    next_stride, next_bound = next(access_iter)
                    if applied_stride != next_stride:
                        # next stride of 0 is allowed in case of broadcasting, but then the
                        # next stride should be forced to 0
                        if next_stride == 0 and StreamerOpts.HasBroadcast in streamers[operand].opts:
                            stride, bound = (0, next_bound // applied_bound)
                        else:
                            raise RuntimeError("Non-contiguous access is not possible for this streamer configuration")
                    else:
                        stride, bound = (
                            applied_stride * applied_bound,
                            next_bound // applied_bound,
                        )
                else:
                    raise NotImplementedError()

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

        new_inputs, new_outputs, new_stride_patterns, ops_to_add = accelerator_type.set_stride_patterns(
            op, snax_stride_patterns
        )
        snax_stride_patterns = list(new_stride_patterns)

        snax_stride_patterns = [pattern.canonicalize() for pattern in snax_stride_patterns]

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
