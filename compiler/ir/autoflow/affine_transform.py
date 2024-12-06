from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass(frozen=True)
class AffineTransform:
    """
    An affine transform mirroring the functionality of xDSLs and MLIRs
    AffineMap, but represented in matrix form to make life much easier.
    This is possible if you don't have to support floordiv/ceildiv operations.
    """

    _A: np.ndarray  # Transformation matrix
    _b: np.ndarray  # Translation vector

    def __post_init__(self):
        # Validate dimensions
        if self._A.ndim != 2:
            raise ValueError("Matrix A must be 2-dimensional.")
        if self._b.ndim != 1:
            raise ValueError("Vector b must be 1-dimensional.")
        if self._A.shape[0] != self._b.shape[0]:
            raise ValueError("Matrix A and vector b must have compatible dimensions.")

    @property
    def num_dims(self) -> int:
        return self._A.shape[0]

    @property
    def num_results(self) -> int:
        return self._A.shape[1]

    def eval(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the affine transformation to a vector or a set of vectors.
        """
        if x.ndim == 1:  # Single vector
            if x.shape[0] != self._A.shape[1]:
                raise ValueError("Input vector x must have a dimension matching the number of columns in A.")
            return self._A @ x + self._b
        elif x.ndim == 2:  # Batch of vectors
            if x.shape[1] != self._A.shape[1]:
                raise ValueError("Input vectors in batch must have a dimension matching the number of columns in A.")
            return (self._A @ x.T).T + self._b
        else:
            raise ValueError("Input x must be 1D (vector) or 2D (batch of vectors).")

    def compose(self, other: Self) -> Self:
        """
        Combine this affine transformation with another.
        The result represents the application of `other` followed by `self`.
        """
        if self._A.shape[1] != other._A.shape[0]:
            raise ValueError("Matrix dimensions of the transformations do not align for composition.")
        new_A = self._A @ other._A
        new_b = self._A @ other._b + self._b
        return type(self)(new_A, new_b)

    def __str__(self):
        return f"AffineTransform(A=\n{self._A},\nb={self._b})"
