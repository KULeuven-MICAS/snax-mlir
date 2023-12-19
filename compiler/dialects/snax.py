from __future__ import annotations

from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    ParameterDef,
    irdl_op_definition,
    irdl_attr_definition,
)
from xdsl.dialects.builtin import ParametrizedAttribute, ArrayAttr, IntAttr, NoneAttr
from compiler.util.TiledStridedLayout import TiledStridedLayout, TiledStride, Stride
from collections.abc import Sequence

# import xdsl.dialects.memref
# from xdsl.dialects.builtin import ArrayAttr, IntAttr


@irdl_attr_definition
class TiledStridedLayoutAttr(ParametrizedAttribute):
    """
    An attribute representing a tiled strided layout of a shaped type.
    """

    name = "tsl"

    strides: ParameterDef[ArrayAttr[ArrayAttr[IntAttr | NoneAttr]]]
    bounds: ParameterDef[ArrayAttr[ArrayAttr[IntAttr | NoneAttr]]]
    offset: ParameterDef[IntAttr | NoneAttr]

    def __init__(
        self,
        strides: ArrayAttr[ArrayAttr[IntAttr | NoneAttr]]
        | Sequence[Sequence[int | None | IntAttr | NoneAttr]],
        bounds: ArrayAttr[ArrayAttr[IntAttr | NoneAttr]]
        | Sequence[Sequence[int | None | IntAttr | NoneAttr]],
        offset: int | None | IntAttr | NoneAttr = 0,
    ):
        if not isinstance(strides, ArrayAttr):
            strides_values: list[ArrayAttr[IntAttr | NoneAttr]] = []
            for tiled_stride in strides:
                tstrides_values: list[IntAttr | NoneAttr] = []
                for stride in tiled_stride:
                    if isinstance(stride, int):
                        tstrides_values.append(IntAttr(stride))
                    elif stride is None:
                        tstrides_values.append(NoneAttr())
                    else:
                        tstrides_values.append(stride)
                tstrides = ArrayAttr(tstrides_values)
                strides_values.append(tstrides)
            strides = ArrayAttr(strides_values)

        if not isinstance(bounds, ArrayAttr):
            bounds_values: list[ArrayAttr[IntAttr | NoneAttr]] = []
            for tiled_stride in bounds:
                tbounds_values: list[IntAttr | NoneAttr] = []
                for stride in tiled_stride:
                    if isinstance(stride, int):
                        tbounds_values.append(IntAttr(stride))
                    elif stride is None:
                        tbounds_values.append(NoneAttr())
                    else:
                        tbounds_values.append(stride)
                tbounds = ArrayAttr(tbounds_values)
                bounds_values.append(tbounds)
            bounds = ArrayAttr(bounds_values)

        if isinstance(offset, int):
            offset = IntAttr(offset)
        if offset is None:
            offset = NoneAttr()

        super().__init__([strides, bounds, offset])

    @staticmethod
    def from_tsl(
        self, tsl: TiledStridedLayout, offset: int | None | IntAttr | NoneAttr = 0
    ) -> "TiledStridedLayoutAttr":
        strides = [[s.tile.stride, s.block.stride] for s in tsl.strides]
        bounds = [[s.tile.bound, s.block.bound] for s in tsl.strides]
        return TiledStridedLayoutAttr(strides, bounds, offset)

    def tsl(self) -> TiledStridedLayout:
        """Construct a TiledStridedLayout object based
        on the TiledStridedLayoutAttr"""
        strides = [
            (stride.data[0].data, stride.data[1].data) for stride in self.strides.data
        ]
        bounds = [
            (bound.data[0].data, bound.data[1].data) for bound in self.bounds.data
        ]

        tsl_strides: list[TiledStride] = []
        for i in range(len(strides)):
            tileStride = Stride(strides[i][0], bounds[i][0])
            blockStride = Stride(strides[i][1], bounds[i][1])
            tsl_strides.append(TiledStride(blockStride, tileStride))
        return TiledStridedLayout(tsl_strides)


# @irdl_attr_definition
# class MemRefType(xdsl.dialects.memref.MemRefType):
#     name = "memref"
#     pass
#     # stridedlayout: ParameterDef[TiledStridedLayoutAttr |
# StridedLayoutAttr | NoneAttr]

#     # @staticmethod
#     # def get_type(
#     #     layout: Attribute
#     # ):
#     #     shape = ArrayAttr[AnyIntegerAttr](
#     #         [
#     #             d if isinstance(d, IntegerAttr) else
# IntegerAttr.from_index_int_value(d)
#     #             for d in [2, 4]
#     #         ]
#     #     )
#     #     return MemRefType ([shape, i32, layout,]

#     #     )


@irdl_op_definition
class ClusterSyncOp(IRDLOperation):
    """Cluster sync operation for a snax cluster. This
    translates directly to the C function snrt_cluster_hw_barrier()"""

    name = "snax.cluster_sync_op"


Snax = Dialect("snax", [ClusterSyncOp], [])
