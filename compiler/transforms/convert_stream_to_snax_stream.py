from dataclasses import dataclass

import numpy as np
from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, memref
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, MemRefType
from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.accelerators.registry import AcceleratorRegistry
from compiler.accelerators.snax import SNAXStreamer
from compiler.accelerators.util import find_accelerator_op
from compiler.dialects import snax_stream, stream
from compiler.ir.autoflow.affine_transform import AffineTransform
from compiler.ir.stream import Schedule, SchedulePattern, scheduler
from compiler.ir.stream.access_pattern import Template


def get_accelerator_info(op: stream.StreamingRegionOpBase) -> Template:
    assert op.accelerator is not None

    # Go and fetch the accelerator op
    accelerator_str = op.accelerator.data
    acc_op = find_accelerator_op(op, accelerator_str)

    if not acc_op:
        raise RuntimeError("AcceleratorOp not found!")

    # get template and template_bounds
    accelerator_type = AcceleratorRegistry().get_acc_info(acc_op)
    assert issubclass(accelerator_type, SNAXStreamer)

    template = accelerator_type.get_template(op)

    return template


@dataclass
class AutoflowScheduler(RewritePattern):
    """
    A pass to convert streaming region operations to schedules.

    Here, the operation is scheduled to an accelerator according to the accelerator template.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: stream.StreamingRegionOp, rewriter: PatternRewriter
    ):
        template = get_accelerator_info(op)

        # Make sure the operands are memrefs
        for memref_operand in op.operands:
            if not isinstance(memref_operand.type, builtin.MemRefType):
                return

        # First, run the stream scheduling algorithm
        schedule_bounds = tuple(op.get_static_pattern_bounds())
        schedule = Schedule(
            SchedulePattern(schedule_bounds, pattern.data)
            for pattern in op.patterns.data
        )
        schedule = scheduler(template, schedule)

        schedule_op = stream.ScheduleOp(
            op.inputs,
            op.outputs,
            ArrayAttr([AffineMapAttr(s.pattern.to_affine_map()) for s in schedule]),
            rewriter.move_region_contents_to_new_regions(op.body),
            schedule[0].bounds,
            [[]],
            op.accelerator,
            op.result_types,
        )

        rewriter.replace_matched_op(schedule_op)


@dataclass
class LayoutResolution(RewritePattern):
    """
    Applies layout resolution by converting a ScheduleOp (mapping the iteration
    space to the operand index space) into an AccessOp (mapping the iteration space
    to memory), using a certain memory layout of the memref operands of the operation.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stream.ScheduleOp, rewriter: PatternRewriter):
        bounds = [x.value.data for x in op.bounds.data]
        schedule = Schedule(
            SchedulePattern(bounds, pattern.data) for pattern in op.patterns
        )

        # small function to generate a list of n zeros with the i-th element 1
        # for example n = 4, i = 1  -> [0, 1, 0, 0]
        def generate_one_list(n: int, i: int):
            return [1 if j == i else 0 for j in range(n)]

        access_patterns: list[AffineMap] = []

        # Do this for every operand:
        for operand in range(len(op.operands)):
            # Mapping from data to memory:
            assert isinstance(memref_type := op.operands[operand].type, MemRefType)

            # Mapping from data to memory:
            data_mem_map: AffineMap = memref_type.get_affine_map_in_bytes()

            # Mapping from access to data:
            access_data_map: AffineMap = schedule[operand].pattern.to_affine_map()

            # Mapping from access to memory:
            access_mem_map: AffineMap = data_mem_map.compose(access_data_map)

            # Make sure no symbols are used (not supported yet)
            if access_mem_map.num_symbols != 0:
                raise RuntimeError(
                    "Access patterns with symbols are not supported yet."
                )

            strides: list[int] = []

            for i in range(access_mem_map.num_dims):
                strides.append(
                    access_mem_map.eval(
                        generate_one_list(access_mem_map.num_dims, i), ()
                    )[0]
                )

            access_patterns.append(
                AffineTransform(np.array([strides]), np.array([0])).to_affine_map()
            )

        new_inputs: list[Operation] = [
            memref.ExtractAlignedPointerAsIndexOp.get(input) for input in op.inputs
        ]
        new_outputs = [
            memref.ExtractAlignedPointerAsIndexOp.get(output) for output in op.outputs
        ]

        new_patterns = ArrayAttr([AffineMapAttr(map) for map in access_patterns])

        access_pattern_op = stream.AccessPatternOp(
            new_inputs,
            new_outputs,
            new_patterns,
            rewriter.move_region_contents_to_new_regions(op.body),
            op.bounds,
            op.accelerator,
            op.result_types,
        )
        rewriter.replace_matched_op(
            [*new_inputs, *new_outputs, access_pattern_op], access_pattern_op.results
        )


@dataclass
class ConvertStreamToSnaxStreamPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stream.AccessPatternOp, rewriter: PatternRewriter):
        template = get_accelerator_info(op)

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
                # ops_to_add.append(
                #         ptr := memref.ExtractAlignedPointerAsIndexOp.get(op.inputs[-1])
                # )
                new_inputs.append(op.inputs[-1])

                # insert zero pattern for C, using the same pattern as D32 but pointing to zero
                # this way, the bias used by the gemm is just a bunch of zeros
                snax_stride_patterns.insert(
                    3,
                    snax_stream.StridePattern(
                        upper_bounds=snax_stride_patterns[3].upper_bounds,
                        temporal_strides=[0]
                        * len(snax_stride_patterns[3].upper_bounds),
                        spatial_strides=[8],
                    ),
                )

                # point C to c0
                ops_to_add.append(
                    # zero pointer will generate 0 values
                    ptr := arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
                )
                new_inputs.append(ptr.result)
            elif len(snax_stride_patterns) == 4:
                # gemm
                #
                # for a gemm, the 8bit-output port D8 are unused, so we create
                # empty patterns for them here
                snax_stride_patterns.insert(2, empty_pattern)
                # ops_to_add.insert(
                #         2, ptr := memref.ExtractAlignedPointerAsIndexOp.get(op.inputs[-1])
                # )
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
class ConvertStreamToSnaxStream(ModulePass):
    name = "convert-stream-to-snax-stream"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(AutoflowScheduler()).rewrite_module(op)
        PatternRewriteWalker(LayoutResolution()).rewrite_module(op)
        PatternRewriteWalker(ConvertStreamToSnaxStreamPattern()).rewrite_module(op)
