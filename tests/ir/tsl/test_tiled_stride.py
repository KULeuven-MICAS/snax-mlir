import pytest

from compiler.ir.tsl.stride import Stride
from compiler.ir.tsl.tiled_stride import TiledStride


@pytest.fixture()
def example_strides():
    return (Stride(1, 4), Stride(4, 6), Stride(24, 2), Stride(None, None))


@pytest.fixture()
def example_tiled_strides(example_strides):
    stride1, stride2, stride3, dynamic_stride = example_strides
    tiledStride1 = TiledStride([stride2, stride1])
    tiledStride2 = TiledStride([stride3, stride2, stride1])
    tiledStride3 = TiledStride([dynamic_stride, stride1])
    return tiledStride1, tiledStride2, tiledStride3


def test_tiled_stride_constructor(example_strides, example_tiled_strides):
    stride1, stride2, stride3, _ = example_strides
    tiledStride1, tiledStride2, _ = example_tiled_strides
    assert tiledStride1.strides[0] == stride2
    assert tiledStride1.strides[1] == stride1
    assert tiledStride2.strides[0] == stride3
    assert tiledStride2.strides[1] == stride2
    assert tiledStride2.strides[2] == stride1


def test_tiled_stride_depth(example_tiled_strides):
    tiledStride1, tiledStride2, tiledStride3 = example_tiled_strides
    assert tiledStride1.depth() == 2
    assert tiledStride2.depth() == 3
    assert tiledStride3.depth() == 2


def test_tiled_stride_str(example_tiled_strides):
    tiledStride1, tiledStride2, tiledStride3 = example_tiled_strides
    assert str(tiledStride1) == "[6, 4] -> (4, 1)"
    assert str(tiledStride2) == "[2, 6, 4] -> (24, 4, 1)"
    assert str(tiledStride3) == "[?, 4] -> (?, 1)"


def test_tiled_stride_iter(example_strides, example_tiled_strides):
    stride1, stride2, stride3, _ = example_strides
    strides = [stride3, stride2, stride1]

    _, tiledStride2, _ = example_tiled_strides

    for depth, stride in tiledStride2:
        assert isinstance(depth, int)
        assert isinstance(stride, Stride)
        assert stride == strides[depth]
