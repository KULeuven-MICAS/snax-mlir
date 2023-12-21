from __future__ import annotations

from xdsl.ir import Dialect
from xdsl.irdl import (
    ParameterDef,
    irdl_attr_definition,
)
from xdsl.dialects.builtin import ParametrizedAttribute, IntAttr, ArrayAttr
from typing import List, Tuple, Iterator
from collections.abc import Iterable
import numpy as np


@irdl_attr_definition
class StrideAttr(ParametrizedAttribute):
    name = "stride"

    stride: ParameterDef[IntAttr]
    bound: ParameterDef[IntAttr]

    def __init__(self, stride: int, bound: int):
        assert stride > 0
        assert bound > 0

        stride = IntAttr(stride)
        bound = IntAttr(bound)

        super().__init__([stride, bound])

    def all_values(self) -> List[int]:
        return list(range(0, self.stride.data * self.bound.data, self.stride.data))

    def __str__(self) -> str:
        return f"{self.stride.data} x {self.bound.data}"


@irdl_attr_definition
class TiledStrideAttr(ParametrizedAttribute):
    name = "tiledStride"

    strides: ParameterDef[ArrayAttr[StrideAttr]]

    def __init__(self, strides: Iterable[StrideAttr]):
        strides_attr: ArrayAttr[StrideAttr] = ArrayAttr(strides)
        super().__init__([strides_attr])

    def get_stride(self, depth: int) -> StrideAttr | None:
        if depth < 0 or depth >= len(self.strides.data):
            return None
        return self.strides.data[depth]

    def __str__(self) -> str:
        strides = ", ".join(str(stride.stride.data) for stride in self.strides.data)
        bounds = ", ".join(str(stride.bound.data) for stride in self.strides.data)
        return f"[{strides}] x [{bounds}]"

    def __iter__(self) -> Iterator[List[int, StrideAttr]]:
        return iter(zip(range(self.depth()), self.strides.data))

    def depth(self) -> int:
        return len(self.strides.data)


@irdl_attr_definition
class TiledStridedLayoutAttr(ParametrizedAttribute):
    name = "tsl"

    tstrides: ParameterDef[ArrayAttr[TiledStrideAttr]]

    def __init__(self, tstrides: List[TiledStrideAttr]):
        tstrides_attr: ArrayAttr[TiledStrideAttr] = ArrayAttr(tstrides)
        super().__init__([tstrides_attr])

    def __str__(self) -> str:
        return "(" + ", ".join(map(str, self.tstrides.data)) + ")"

    def __iter__(self) -> Iterator[Tuple[int, int, StrideAttr]]:
        # append dimension to iterators of tstrides
        result = [
            list(map(lambda x: (dim,) + x, iter(tsride)))
            for dim, tsride in zip(range(self.dimension()), self.tstrides.data)
        ]
        # unpack nested list
        result = [x for y in result for x in y]
        # return result
        return iter(result)

    def dimension(self):
        return len(self.tstrides.data)

    def get_stride(self, dim: int, depth: int):
        return self.tstrides.data[dim].strides.data[depth]

    def all_values(self) -> np.ndarray:
        # return a set of all the elements
        # in the iteration space
        result = np.array([0])

        for _, _, stride in self:
            next_stride = np.array(stride.all_values())
            # for every stride, add a dimension and broadcast sum
            result = np.squeeze(
                np.expand_dims(result, -1) + np.expand_dims(next_stride, 0)
            )

        # flatten multi-dimensional array
        result = result.flatten()
        return result

    def self_overlaps(self) -> bool:
        # check for duplicates in the iteration space
        (
            u,
            c,
        ) = np.unique(self.all_values(), return_counts=True)
        dup = u[c > 1]  # get duplicates
        return len(dup) > 0  # return True if there are duplicates

    def is_dense(self) -> bool:
        # check for gaps in the iteration space
        if self.self_overlaps():
            return False

        all_values = self.all_values()
        return np.max(all_values) == len(all_values) - 1

    def largest_common_contiguous_block(
        self, other: "TiledStridedLayoutAttr"
    ) -> List[StrideAttr]:
        result: List[StrideAttr] = []

        # does not work on illegal workloads
        if self.self_overlaps():
            return result
        if other.self_overlaps():
            return result

        # find largest contiguous block, starting with stride = 1
        current_stride = 1
        while True:
            # search for contiguous block
            next_stride = next(
                (
                    (dim, depth, stride_self)
                    for dim, depth, stride_self in self
                    if stride_self.stride.data == current_stride
                ),
                None,
            )

            # check if contiguous block is found
            if next_stride is None:
                return result
            else:
                dim, depth, stride_self = next_stride

            # check if contiguous block is common with other layout
            stride_other = other.tstrides.data[dim].strides.data[depth]
            if stride_self == stride_other:
                result.append(stride_self)
                current_stride = stride_self.stride.data * stride_self.bound.data

            else:
                return result

    def get_2d_dma_strategy(self, other: "TiledStridedLayoutAttr"):
        # assumptions for now:
        #   tile sizes are equal between source and target

        # find largest common contiguous block to copy
        lcb = self.largest_common_contiguous_block(other)

        # other strides are handled in software, sourt by bound
        remaining_strides = [x for x in self if x not in lcb]
        remaining_strides: List[StrideAttr] = sorted(
            remaining_strides, key=lambda x: x[2].bound.data
        )

        if len(remaining_strides) == 0:
            # TODO
            pass
        elif len(remaining_strides) == 1:
            # TODO
            pass
        else:
            dma_loop = remaining_strides[0]
            remaining_strides[1:]

        other.get_stride(dma_loop[0], dma_loop[1])

        pass
        # other stuff TODO (see older commit for util implementation)
        # ... but this was incorrect


TSL = Dialect("tsl", [], [])
