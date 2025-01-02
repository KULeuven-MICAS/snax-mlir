import numpy as np
import pytest

from compiler.ir.autoflow import AffineTransform
from compiler.util.canonicalize_affine import canonicalize_map


def test_affine_transform_initialization_valid():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([7, 8])
    transform = AffineTransform(A, b)
    assert np.array_equal(transform.A, A)
    assert np.array_equal(transform.b, b)
    assert transform.num_dims == 3
    assert transform.num_results == 2


def test_affine_transform_initialization_invalid_dimensions():
    A = np.array([[1, 2], [3, 4]])
    b = np.array([5])  # Incompatible size
    with pytest.raises(
        ValueError, match="Matrix A and vector b must have compatible dimensions."
    ):
        AffineTransform(A, b)


def test_affine_transform_eval_single_vector():
    A = np.array([[1, 0], [0, 1]])
    b = np.array([1, 2])
    transform = AffineTransform(A, b)
    x = np.array([3, 4])
    result = transform.eval(x)
    expected = np.array([4, 6])
    assert np.array_equal(result, expected)


def test_affine_transform_eval_batch_of_vectors():
    A = np.array([[1, 0], [0, 1]])
    b = np.array([1, 2])
    transform = AffineTransform(A, b)
    x_batch = np.array([[3, 4], [5, 6]])
    result = transform.eval(x_batch)
    expected = np.array([[4, 6], [6, 8]])
    assert np.array_equal(result, expected)


def test_affine_transform_eval_invalid_vector_dimension():
    A = np.array([[1, 0], [0, 1]])
    b = np.array([1, 2])
    transform = AffineTransform(A, b)
    x = np.array([3])  # Incompatible dimension
    with pytest.raises(
        ValueError,
        match="Input vector x must have a dimension matching the number of columns in A.",
    ):
        transform.eval(x)


def test_affine_transform_compose():
    A1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([5, 6])
    transform1 = AffineTransform(A1, b1)

    A2 = np.array([[0, 1], [1, 0]])
    b2 = np.array([7, 8])
    transform2 = AffineTransform(A2, b2)

    composed = transform1.compose(transform2)

    expected_A = A1 @ A2
    expected_b = A1 @ b2 + b1

    assert np.array_equal(composed.A, expected_A)
    assert np.array_equal(composed.b, expected_b)


def test_affine_transform_compose_invalid_dimensions():
    A1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([5, 6])
    transform1 = AffineTransform(A1, b1)

    A2 = np.array([[1, 0, 0], [0, 1, 0]])
    b2 = np.array([7, 8])
    transform2 = AffineTransform(A2, b2)

    with pytest.raises(
        ValueError,
        match="Matrix dimensions of the transformations do not align for composition.",
    ):
        transform2.compose(transform1)


def test_affine_transform_str():
    A = np.array([[1, 0], [0, 1]])
    b = np.array([1, 2])
    transform = AffineTransform(A, b)
    expected = "AffineTransform(A=\n[[1 0]\n [0 1]],\nb=[1 2])"
    assert str(transform) == expected


def test_affine_map_interop():
    map = AffineMap.from_callable(lambda a, b, c: (a + 2 * b, -b + c, 3 * a + c + 4))

    # convert AffineMap to AffineTransform
    transform = AffineTransform.from_affine_map(map)

    expected_a = np.array([[1, 2, 0], [0, -1, 1], [3, 0, 1]])
    expected_b = np.array([0, 0, 4])

    assert (transform.A == expected_a).all()
    assert (transform.b == expected_b).all()

    # convert back to AffineMap

    original_map = transform.to_affine_map()
    assert canonicalize_map(map) == canonicalize_map(original_map)
