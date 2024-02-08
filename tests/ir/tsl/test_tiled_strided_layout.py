import pytest

from compiler.ir.tsl.stride import Stride
from compiler.ir.tsl.tiled_stride import TiledStride
from compiler.ir.tsl.tiled_strided_layout import TiledStridedLayout


@pytest.fixture()
def example_tsl():
    tiledStride1 = TiledStride(
        [
            Stride(32, 2),
            Stride(4, 4),
        ]
    )
    tiledStride2 = TiledStride(
        [
            Stride(16, 2),
            Stride(1, 4),
        ]
    )
    tiledStride3 = TiledStride(
        [
            Stride(None, None),
            Stride(1, 4),
        ]
    )
    tsl = TiledStridedLayout([tiledStride1, tiledStride2], offset=5)
    tsl2 = TiledStridedLayout([tiledStride1, tiledStride3], offset=7)
    return tsl, tsl2


def test_tsl_constructor(example_tsl):
    tsl, _ = example_tsl
    assert isinstance(tsl.tstrides[0], TiledStride)
    assert isinstance(tsl.tstrides[1], TiledStride)


def test_tsl_from_strides():
    strides = [None, 1]
    tile_bounds = [[16, 4], [16, 4]]
    tsl_constructor = TiledStridedLayout(
        [
            TiledStride([Stride(None, 16), Stride(None, 4)]),
            TiledStride([Stride(4, 16), Stride(1, 4)]),
        ]
    )
    tsl_from_strides = TiledStridedLayout.from_strides(strides, tile_bounds)
    assert tsl_constructor == tsl_from_strides


def test_tsl_str(example_tsl):
    tsl, tsl2 = example_tsl
    assert str(tsl) == "[2, 4] -> (32, 4), [2, 4] -> (16, 1), offset: 5"
    assert str(tsl2) == "[2, 4] -> (32, 4), [?, 4] -> (?, 1), offset: 7"


def test_tsl_iter(example_tsl):
    tsl, _ = example_tsl
    count = 0
    for dim, depth, stride in tsl:
        count += 1
        assert isinstance(dim, int)
        assert isinstance(depth, int)
        assert isinstance(stride, Stride)
        assert stride == tsl.get_stride(dim, depth)
    assert count == tsl.dimension() * tsl.tstrides[0].depth()


def test_tsl_all_values(example_tsl):
    tsl, tsl2 = example_tsl
    assert set(tsl.all_values()) == set(range(64))
    with pytest.raises(ValueError):
        tsl2.all_values()


def test_tsl_tile_bounds(example_tsl):
    tsl, _ = example_tsl
    assert tsl.tile_bounds() == [[2, 4], [2, 4]]


def test_tsl_self_overlaps(example_tsl):
    tsl, _ = example_tsl
    assert not tsl.self_overlaps()

    tiledStride1 = TiledStride(
        [
            Stride(16, 2),
            Stride(4, 4),
        ]
    )
    tiledStride2 = TiledStride(
        [
            Stride(16, 2),
            Stride(1, 4),
        ]
    )
    tsl2 = TiledStridedLayout([tiledStride1, tiledStride2])

    assert tsl2.self_overlaps()


def test_tsl_is_dense(example_tsl):
    tsl, _ = example_tsl
    assert tsl.is_dense()

    tiledStride1 = TiledStride(
        [
            Stride(64, 2),
            Stride(4, 4),
        ]
    )
    tiledStride2 = TiledStride(
        [
            Stride(16, 2),
            Stride(1, 4),
        ]
    )
    tsl2 = TiledStridedLayout([tiledStride1, tiledStride2])

    assert not tsl2.is_dense()


def test_tsl_equal_tile_bounds(example_tsl):
    tsl, tsl2 = example_tsl
    assert tsl.equal_tile_bounds(tsl)
    assert not tsl.equal_tile_bounds(tsl2)


def test_tsl_largest_common_contiguous_block():
    tiledStride1 = TiledStride(
        [
            Stride(16, 2),
            Stride(4, 4),
        ]
    )
    tiledStride2 = TiledStride(
        [
            Stride(32, 2),
            Stride(1, 4),
        ]
    )
    tsl1 = TiledStridedLayout([tiledStride1, tiledStride2])
    tiledStride1 = TiledStride(
        [
            Stride(32, 2),
            Stride(4, 4),
        ]
    )
    tiledStride2 = TiledStride(
        [
            Stride(16, 2),
            Stride(1, 4),
        ]
    )
    tsl2 = TiledStridedLayout([tiledStride1, tiledStride2])
    tiledStride1 = TiledStride(
        [
            Stride(32, 2),
            Stride(8, 4),
        ]
    )
    tiledStride2 = TiledStride(
        [
            Stride(4, 2),
            Stride(1, 4),
        ]
    )
    tsl3 = TiledStridedLayout([tiledStride1, tiledStride2])

    ## Check largest common contiguous block between tsl1 and tsl2
    lccb1 = tsl1.largest_common_contiguous_block(tsl2)
    assert len(lccb1) == 2
    assert lccb1[0].step == 1
    assert lccb1[0].bound == 4
    assert lccb1[1].step == 4
    assert lccb1[1].bound == 4

    ## other way around should be the same
    lccb2 = tsl2.largest_common_contiguous_block(tsl1)
    assert len(lccb2) == 2
    assert lccb2[0].step == 1
    assert lccb2[0].bound == 4
    assert lccb2[1].step == 4
    assert lccb2[1].bound == 4

    lccb3 = tsl1.largest_common_contiguous_block(tsl3)
    assert len(lccb3) == 1
    assert lccb3[0].step == 1
    assert lccb3[0].bound == 4
