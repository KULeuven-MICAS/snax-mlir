from dataclasses import dataclass
from math import ceil, prod

import numpy as np
from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.ir import Attribute, SSAValue
from xdsl.parser import MemRefType
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa

from snaxc.accelerators.acc_context import AccContext
from snaxc.accelerators.snax import SNAXStreamer
from snaxc.dialects import dart
from snaxc.dialects.snax import LayoutCast
from snaxc.dialects.tsl import TiledStridedLayoutAttr
from snaxc.ir.dart.access_pattern import Schedule, SchedulePattern
from snaxc.ir.tsl import Stride, TiledStride, TiledStridedLayout


def spatial_dims(ctx: AccContext, op: dart.ScheduleOp) -> int:
    """
    Get the number of spatial dimensions of the accelerator.
    """
    # get accelerator
    assert op.accelerator
    accelerator_type = ctx.get_acc(op.accelerator.data)
    assert isinstance(accelerator_type, SNAXStreamer)
    template = accelerator_type.get_template(op)
    return template[0].pattern.num_dims


def ensure_access_granularity(
    ctx: AccContext, current_stride: int, schedule_dim: int, op: dart.ScheduleOp, operand: SSAValue
) -> int:
    if current_stride == 1:
        return current_stride

    # assess access granularity:
    # TODO: build out this system to be better
    # this is a hack that works, this information needs to come from accelerators:
    assert isa(operand.type, MemRefType[builtin.FixedBitwidthType])

    if ctx.banked_layout and schedule_dim >= spatial_dims(ctx, op):
        banks = 1024 if ctx.get_acc("snax_gemmx").m == 1 else 256  # pyright: ignore
        temporal_access_granularity = banks if operand.type.get_element_type().bitwidth == 8 else 16  # in elements
        if current_stride % temporal_access_granularity != 0:
            current_stride += (temporal_access_granularity - current_stride) % banks
    elif schedule_dim >= spatial_dims(ctx, op):
        # we are in temporal regime
        temporal_access_granularity = 8 if operand.type.get_element_type().bitwidth == 8 else 16  # in elements
        if current_stride % temporal_access_granularity != 0:
            current_stride += (temporal_access_granularity - current_stride) % 64
    else:
        # we are in spatial regime
        spatial_access_granularity = 8 if operand.type.get_element_type().bitwidth == 8 else 2  # in elements
        if current_stride % spatial_access_granularity != 0:
            current_stride += (spatial_access_granularity - current_stride) % 64

    return current_stride


@dataclass
class AddCyclicMemoryLayout(RewritePattern):
    """
    Automatically generates cyclic memory layouts for the operands of a schedule operation.
    The layout is determined based on access patterns in the schedule.

    - Dimensions accessed in the innermost loop receive the lowest stride.
    - Other dimensions are assigned increasing strides in a contiguous manner.
    - This approach can yield variations of formats such as NCHW, NHWC, etc., optimizing for efficient memory access.

    Additionally, the pass supports tiled layouts, where data is first tiled before assigning contiguous strides.
    This enables efficient memory layouts such as tiled block formats.
    """

    # allow for tiled layouts?
    ctx: AccContext
    tiled_layout: bool = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dart.ScheduleOp, rewriter: PatternRewriter):
        # do not alter pre-existing layouts
        for operand in op.operands:
            if isa(operand.type, MemRefType[Attribute]) and isinstance(operand.type.layout, TiledStridedLayoutAttr):
                return

        # get schedule from op
        bounds = [x.value.data for x in op.bounds]
        schedule = Schedule(SchedulePattern(bounds, x.data) for x in op.patterns)

        # list to store newly generated operands for op
        new_operands: list[LayoutCast] = []

        # a separate layout is determined for every operand
        for operand, schedule in zip(op.operands, schedule):
            assert isa(memref_type := operand.type, MemRefType[Attribute])

            # start assigning contiguous, starting from stride = 1
            current_stride = 1

            # create a list to keep strides for every dimension.
            # for non-tiled layouts, every dimension will be assigned 1 stride
            # for tiled layouts, every dimension can be assigned multiple strides
            strides: list[list[Stride]] = [[] for _ in range(memref_type.get_num_dims())]

            # iterate over the columns of the schedule pattern in reversed order, to find out
            # which dimension is accessed in the innermost loop of the operation
            for schedule_dim, (schedule_bound, accesses) in enumerate(
                zip(schedule.bounds[::-1], np.flip(schedule.pattern.A, axis=1).T)
            ):
                # normalize accesses to binary list
                # this list will now have a 1 at the index of the dimension that is accessed
                accesses = tuple(0 if x == 0 else 1 for x in accesses)

                if 1 not in accesses:
                    continue

                current_stride = ensure_access_granularity(self.ctx, current_stride, schedule_dim, op, operand)

                # find operand dimension that is accessed
                accessed_dim = accesses.index(1)

                # we have determined the dimension and step for this layout
                # now we must determine the bound.
                # For non-tiled layouts, this bound will be equal to the operand shape.
                # For tiled layouts, the bound is set equal to the schedule bound.

                # the existing bound of the current layout
                existing_bound = prod(s.bound for s in strides[accessed_dim] if s.bound)

                # the remaining size of the operand dimension
                size_remaining = ceil(memref_type.get_shape()[accessed_dim] // existing_bound)

                # can we further tile the layout according to the remaining size?
                to_tile = self.tiled_layout

                if to_tile:
                    # only apply tiling if entire size is nicely divisible by the tile size for now
                    # (this isn't strictly necessary but might break some things)
                    if size_remaining % schedule_bound != 0:
                        to_tile = False

                    # only apply tiling if all access patterns remain hyperrectangular
                    # for example if there is one dimensions that accesses with stride=1, bound=3 and another dim with
                    # stride=1, bound=8
                    # we cannot tile for either 8 or 3 because then the other pattern is no longer affine
                    else:
                        for stride, bound in zip(schedule.pattern.A[accessed_dim, :], schedule.bounds):
                            if stride % schedule_bound != 0 and bound != schedule_bound:
                                to_tile = False

                if to_tile:
                    layout_bound = schedule_bound
                else:
                    layout_bound = size_remaining

                # assign this current stride to the relevant operand dimension
                strides[accesses.index(1)].insert(0, Stride(current_stride, layout_bound))

                # increase current stride
                current_stride = current_stride * layout_bound

            # fill up empty strides
            for stride in strides:
                if not len(stride):
                    stride.append(Stride(current_stride, 1))

            layout = TiledStridedLayout([TiledStride(s) for s in strides]).canonicalize()
            tsl = TiledStridedLayoutAttr(layout)

            new_operands.append(LayoutCast.from_type_and_target_layout(operand, tsl))

        rewriter.insert_op(new_operands, InsertPoint.before(op))
        for i, new_operand in enumerate(new_operands):
            op.operands[i] = new_operand.dest


@dataclass(frozen=True)
class SetMemoryLayout(ModulePass):
    name = "set-memory-layout"

    tiled: bool | None = True

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        assert isinstance(ctx, AccContext)
        tiled = ctx.layout in ("cyclic", "banked")
        # tiled = self.tiled if self.tiled is not None else True
        PatternRewriteWalker(AddCyclicMemoryLayout(tiled_layout=tiled, ctx=ctx)).rewrite_module(op)
