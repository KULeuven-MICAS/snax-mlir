from __future__ import annotations

from dataclasses import dataclass
from compiler.ir.tsl.stride import Stride
from compiler.ir.tsl.tiled_stride import TiledStride
from typing import Iterator, List, Tuple
import numpy as np


@dataclass
class TiledStridedLayout:
    """TiledStridedLayout is a collection of TiledStrides to represent a tiled
    strided layout for a multi-dimensional array.

    Args:
        tstrides (List[TiledStride]): A list of TiledStrides, one for each dimension
    """

    tstrides: List[TiledStride]

    def __init__(self, tstrides: List[TiledStride]):
        self.tstrides = tstrides

    def __str__(self) -> str:
        return "(" + ", ".join(map(str, self.tstrides)) + ")"

    def __iter__(self) -> Iterator[Tuple[int, int, Stride]]:
        """Returns an iterator over the dimensions, depths and
        strides of the Tiled Strided Layout"""

        result = [
            list(map(lambda x: (dim,) + x, iter(tsride)))
            for dim, tsride in zip(range(self.dimension()), self.tstrides)
        ]
        # unpack nested list
        result = [x for y in result for x in y]
        # return result
        return iter(result)

    def dimension(self) -> int:
        """Get the number of dimensions in the Tiled Strided Layout"""
        return len(self.tstrides)

    def get_stride(self, dim: int, depth: int) -> Stride:
        """Get the stride at a particular dimensaion and depth of
        the Tiled Strided Layout"""
        return self.tstrides[dim].strides[depth]

    def all_values(self) -> np.ndarray:
        """
        Returns a numpy array containing all the elements in the iteration space.
        """
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
        """Check if the Tiled Strided Layout contains overlapping elements"""
        (
            u,
            c,
        ) = np.unique(self.all_values(), return_counts=True)
        dup = u[c > 1]  # get duplicates
        return len(dup) > 0  # return True if there are duplicates

    def is_dense(self) -> bool:
        """Check if the Tiled Strided Layout contains no gaps"""
        if self.self_overlaps():
            return False

        all_values = self.all_values()
        return np.max(all_values) == len(all_values) - 1

    def largest_common_contiguous_block(
        self, other: "TiledStridedLayout"
    ) -> List[Stride]:
        """Get the largest common contiguous block between two Tiled Strided Layouts"""
        result: List[Stride] = []

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
                    if stride_self.stride == current_stride
                ),
                None,
            )

            # check if contiguous block is found
            if next_stride is None:
                return result
            else:
                dim, depth, stride_self = next_stride

            # check if contiguous block is common with other layout
            stride_other = other.get_stride(dim, depth)
            if stride_self == stride_other:
                result.append(stride_self)
                current_stride = stride_self.stride * stride_self.bound

            else:
                return result
