from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stride:
    """A stride represents the distance between elements in an array.
    A Stride is defined by a step and a bound, and can be
    used to generate all values within the bound.

    Args:
        step (int | None): The step of the Stride
            None represents a dynamic step
        bound (int | None): The bound of the Stride
            None represents a dynamic bound
    """

    step: int | None
    bound: int | None

    def is_dynamic(self) -> bool:
        """Check if the Stride is dynamic"""
        return self.step is None or self.bound is None

    def all_values(self) -> list[int]:
        """Get all values within the bound of the Stride"""
        if self.is_dynamic():
            raise ValueError("Cannot get all values of a dynamic stride")
        assert self.step
        assert self.bound
        return list(range(0, self.step * self.bound, self.step))

    def __str__(self) -> str:
        step = "?" if self.step is None else str(self.step)
        bound = "?" if self.bound is None else str(self.bound)
        return f"{bound} -> {step}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stride):
            return NotImplemented
        return self.step == other.step and self.bound == other.bound
