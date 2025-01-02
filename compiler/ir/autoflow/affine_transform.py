from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from typing_extensions import Self


@dataclass(frozen=True)
class AffineTransform:
    """
    An affine transform mirroring the functionality of xDSLs and MLIRs
    AffineMap, but represented in matrix form to make life much easier.
    This is possible if you don't have to support floordiv/ceildiv operations.
    """

    A: npt.NDArray[np.int_]  # Transformation matrix
    b: npt.NDArray[np.int_]  # Translation vector

    def __post_init__(self):
        # Validate dimensions
        if self.A.ndim != 2:
            raise ValueError("Matrix A must be 2-dimensional.")
        if self.b.ndim != 1:
            raise ValueError("Vector b must be 1-dimensional.")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError("Matrix A and vector b must have compatible dimensions.")

    @property
    def num_dims(self) -> int:
        return self.A.shape[0]

    @property
    def num_results(self) -> int:
        return self.A.shape[1]

    def eval(self, x: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """
        Apply the affine transformation to a vector or a set of vectors.
        """
        if x.ndim == 1:  # Single vector
            if x.shape[0] != self.A.shape[1]:
                raise ValueError(
                    "Input vector x must have a dimension matching the number of columns in A."
                )
            return self.A @ x + self.b
        elif x.ndim == 2:  # Batch of vectors
            if x.shape[1] != self.A.shape[1]:
                raise ValueError(
                    "Input vectors in batch must have a dimension matching the number of columns in A."
                )
            return (self.A @ x.T).T + self.b
        else:
            raise ValueError("Input x must be 1D (vector) or 2D (batch of vectors).")

    def compose(self, other: Self) -> Self:
        """
        Combine this affine transformation with another.
        The result represents the application of `other` followed by `self`.
        """
        if self.A.shape[1] != other.A.shape[0]:
            raise ValueError(
                "Matrix dimensions of the transformations do not align for composition."
            )
        new_A = self.A @ other.A
        new_b = self.A @ other.b + self.b
        return type(self)(new_A, new_b)

    def __str__(self):
        return f"AffineTransform(A=\n{self.A},\nb={self.b})"
