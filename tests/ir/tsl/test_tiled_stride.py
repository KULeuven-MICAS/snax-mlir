import pytest

from snaxc.ir.tsl.stride import Stride
from snaxc.ir.tsl.tiled_stride import TiledStride


@pytest.fixture()
def example_strides():
    return (Stride(1, 4), Stride(4, 6), Stride(24, 2), Stride(None, None))


@pytest.fixture()
def example_tiled_strides(example_strides: tuple[Stride, ...]):
    stride1, stride2, stride3, dynamic_stride = example_strides
    tiledStride1 = TiledStride([stride2, stride1])
    tiledStride2 = TiledStride([stride3, stride2, stride1])
    tiledStride3 = TiledStride([dynamic_stride, stride1])
    return tiledStride1, tiledStride2, tiledStride3


def test_tiled_stride_constructor(
    example_strides: tuple[Stride, ...], example_tiled_strides: tuple[TiledStride, ...]
):
    stride1, stride2, stride3, _ = example_strides
    tiledStride1, tiledStride2, _ = example_tiled_strides
    assert tiledStride1.strides[0] == stride2
    assert tiledStride1.strides[1] == stride1
    assert tiledStride2.strides[0] == stride3
    assert tiledStride2.strides[1] == stride2
    assert tiledStride2.strides[2] == stride1


def test_tiled_stride_from_stride():
    tiledStride1 = TiledStride.from_stride(1, [4, 6])
    tiledStride2 = TiledStride.from_stride(24, [2, 6, 4])
    assert tiledStride1.strides[0] == Stride(6, 4)
    assert tiledStride1.strides[1] == Stride(1, 6)
    assert tiledStride2.strides[0] == Stride(24 * 4 * 6, 2)
    assert tiledStride2.strides[1] == Stride(24 * 4, 6)
    assert tiledStride2.strides[2] == Stride(24, 4)


def test_tiled_stride_depth(example_tiled_strides: tuple[TiledStride, ...]):
    tiledStride1, tiledStride2, tiledStride3 = example_tiled_strides
    assert tiledStride1.depth() == 2
    assert tiledStride2.depth() == 3
    assert tiledStride3.depth() == 2


def test_tiled_stride_str(example_tiled_strides: tuple[TiledStride, ...]):
    tiledStride1, tiledStride2, tiledStride3 = example_tiled_strides
    assert str(tiledStride1) == "[6, 4] -> (4, 1)"
    assert str(tiledStride2) == "[2, 6, 4] -> (24, 4, 1)"
    assert str(tiledStride3) == "[?, 4] -> (?, 1)"


def test_tiled_stride_iter(
    example_strides: tuple[Stride, ...], example_tiled_strides: tuple[TiledStride, ...]
):
    stride1, stride2, stride3, _ = example_strides
    strides = [stride3, stride2, stride1]

    _, tiledStride2, _ = example_tiled_strides

    for depth, stride in tiledStride2:
        assert isinstance(depth, int)
        assert isinstance(stride, Stride)
        assert stride == strides[depth]


def test_tiled_stride_tile_bounds(example_tiled_strides: tuple[TiledStride, ...]):
    tiledStride1, tiledStride2, tiledStride3 = example_tiled_strides
    assert tiledStride1.tile_bounds() == [6, 4]
    assert tiledStride2.tile_bounds() == [2, 6, 4]
    assert tiledStride3.tile_bounds() == [None, 4]


def test_tiled_stride_canonicalize():
    # normal tstrides are unaffected
    tstride_normal = TiledStride([Stride(16, 8), Stride(1, 8)])
    assert tstride_normal.canonicalize() == tstride_normal

    # strides with bound 1 are evicted
    tstride_bound1 = TiledStride([Stride(16, 1), Stride(1, 8)])
    assert tstride_bound1.canonicalize() == TiledStride([Stride(1, 8)])

    # squashable strides are squashed
    tstride_squashme = TiledStride([Stride(8, 8), Stride(1, 8)])
    assert tstride_squashme.canonicalize() == TiledStride([Stride(1, 64)])

    # bound 1 + squash
    tstride = TiledStride([Stride(8, 8), Stride(16, 1), Stride(1, 8)])
    assert tstride.canonicalize() == TiledStride([Stride(1, 64)])

    # squash + bound 1
    tstride = TiledStride([Stride(64, 1), Stride(8, 8), Stride(1, 8)])
    assert tstride.canonicalize() == TiledStride([Stride(1, 64)])

    # normal remains the same
    tstride = TiledStride([Stride(8, 2), Stride(16, 8), Stride(1, 8)])
    assert tstride.canonicalize() == tstride
