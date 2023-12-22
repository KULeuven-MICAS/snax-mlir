import pytest
from compiler.ir.tsl.stride import Stride


@pytest.fixture
def example_strides():
    return (Stride(1, 4), Stride(4, 6), Stride(24, 2))


def test_stride_constructor(example_strides):
    stride1, stride2, _ = example_strides
    assert stride1.stride == 1
    assert stride1.bound == 4
    assert stride2.stride == 4
    assert stride2.bound == 6


def test_stride_all_values(example_strides):
    stride1, stride2, _ = example_strides
    assert stride1.all_values() == [0, 1, 2, 3]
    assert stride2.all_values() == [0, 4, 8, 12, 16, 20]


def test_stride_str(example_strides):
    stride1, stride2, stride3 = example_strides
    assert str(stride1) == "1 x 4"
    assert str(stride2) == "4 x 6"
    assert str(stride3) == "24 x 2"
