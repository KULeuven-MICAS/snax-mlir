from __future__ import annotations

from typing import List
from dataclasses import dataclass


@dataclass
class Stride:
    """A stride represents the distance between elements in an array.
    A Stride is defined by a stride and a bound, and the stride can be
    used to generate all values within the bound.

    Args:
        stride (int): The stride of the Stride
        bound (int): The bound of the Stride
    """

    stride: int
    bound: int

    def all_values(self) -> List[int]:
        """Get all values within the bound of the Stride"""
        return list(range(0, self.stride * self.bound, self.stride))

    def __str__(self) -> str:
        return f"{self.stride} x {self.bound}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stride):
            return NotImplemented
        return self.stride == other.stride and self.bound == other.bound
