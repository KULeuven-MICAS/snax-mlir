from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stride:
    """A stride represents the distance between elements in an array.
    A Stride is defined by a stride and a bound, and the stride can be
    used to generate all values within the bound.

    Args:
        stride (int | None): The stride of the Stride
            None represents a dynamic stride
        bound (int | None): The bound of the Stride
            None represents a dynamic bound
    """

    stride: int | None
    bound: int | None

    def is_dynamic(self) -> bool:
        """Check if the Stride is dynamic"""
        return self.stride is None or self.bound is None

    def all_values(self) -> list[int]:
        """Get all values within the bound of the Stride"""
        if self.is_dynamic():
            raise ValueError("Cannot get all values of a dynamic stride")
        return list(range(0, self.stride * self.bound, self.stride))

    def __str__(self) -> str:
        stride = "?" if self.stride is None else str(self.stride)
        bound = "?" if self.bound is None else str(self.bound)
        return f"{stride} x {bound}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stride):
            return NotImplemented
        return self.stride == other.stride and self.bound == other.bound
