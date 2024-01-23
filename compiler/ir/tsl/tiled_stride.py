from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from compiler.ir.tsl.stride import Stride


@dataclass
class TiledStride:
    """
    A Tiled Stride is a collection of Strides, one for each level of a Tiled Stride.

    The class works with an arbitrary number of tile levels. The most common use case
    is one level of tiling, which results in a tile depth of 2.

    For example, consider a Tiled Stride for a vector of length 8 of [2, 4] -> (16, 1).
    This Tiled Stride means that the distance between elements in the first
    tile of size 4 is 1, and the distance between elements in the second tile
    of size 2 is 16. This results in the following memory addresses:

    1 - 2 - 3 - 4 - 16 - 17 - 18 - 19

    """

    strides: list[Stride]

    def __init__(self, strides: list[Stride]):
        self.strides = list(strides)

    def __str__(self) -> str:
        strides = ", ".join(
            str(stride.step) if stride.step else "?" for stride in self.strides
        )
        bounds = ", ".join(
            str(stride.bound) if stride.bound else "?" for stride in self.strides
        )
        return f"[{bounds}] -> ({strides})"

    def __iter__(self) -> Iterator[list[int, Stride]]:
        """Returns an iterator of (depth, stride) over all
            the strides of the Tiled Stride

        Yields:
            Iterator[List[int, Stride]]: An iterator over the depths and strides
            of the Tiled Stride
        """
        yield from zip(range(self.depth()), self.strides)

    def is_dynamic(self) -> bool:
        """Check if the Tiled Stride is dynamic"""
        return any(stride.is_dynamic() for _, stride in self.strides)

    def depth(self) -> int:
        """Get the number of strides in the Tiled Stride

        Returns:
            int: The number of strides in the Tiled Stride.
        """
        return len(self.strides)

    def all_values(self) -> list[list[int]]:
        """Get all resulting values of the Tiled Stride"""
        return [stride.all_values() for stride in self.strides]

    def get_stride(self, depth: int) -> Stride | None:
        """Get the stride at a particular depth of the Tiled Stride

        Args:
            depth (int): The depth of the stride to retrieve.

        Returns:
            Stride | None: The stride at the specified depth,
            or None if the depth is invalid.
        """
        try:
            return self.strides[depth]
        except IndexError:
            return None
