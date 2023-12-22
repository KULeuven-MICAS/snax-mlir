import pytest

from compiler.ir.tsl.stride import Stride
from compiler.ir.tsl.tiled_stride import TiledStride
from compiler.ir.tsl.tiled_strided_layout import TiledStridedLayout


@pytest.fixture()
def example_tsl():
    tiledStride1 = TiledStride(
        [
            Stride(4, 4),
            Stride(32, 2),
        ]
    )
    tiledStride2 = TiledStride(
        [
            Stride(1, 4),
            Stride(16, 2),
        ]
    )
    tsl = TiledStridedLayout([tiledStride1, tiledStride2])
    return tsl


def test_tsl_constructor(example_tsl):
    tsl = example_tsl
    assert isinstance(tsl.tstrides[0], TiledStride)
    assert isinstance(tsl.tstrides[1], TiledStride)


def test_tsl_str(example_tsl):
    tsl = example_tsl
    assert str(tsl) == "([4, 32] * [4, 2], [1, 16] * [4, 2])"


def test_tsl_iter(example_tsl):
    tsl = example_tsl
    count = 0
    for dim, depth, stride in tsl:
        count += 1
        assert isinstance(dim, int)
        assert isinstance(depth, int)
        assert isinstance(stride, Stride)
        assert stride == tsl.get_stride(dim, depth)
    assert count == tsl.dimension() * tsl.tstrides[0].depth()


def test_tsl_all_values(example_tsl):
    tsl = example_tsl
    assert set(tsl.all_values()) == set(range(64))


def test_tsl_self_overlaps(example_tsl):
    tsl = example_tsl
    assert not tsl.self_overlaps()

    tiledStride1 = TiledStride(
        [
            Stride(4, 4),
            Stride(16, 2),
        ]
    )
    tiledStride2 = TiledStride(
        [
            Stride(1, 4),
            Stride(16, 2),
        ]
    )
    tsl2 = TiledStridedLayout([tiledStride1, tiledStride2])

    assert tsl2.self_overlaps()


def test_tsl_is_dense(example_tsl):
    tsl = example_tsl
    assert tsl.is_dense()

    tiledStride1 = TiledStride(
        [
            Stride(4, 4),
            Stride(64, 2),
        ]
    )
    tiledStride2 = TiledStride(
        [
            Stride(1, 4),
            Stride(16, 2),
        ]
    )
    tsl2 = TiledStridedLayout([tiledStride1, tiledStride2])

    assert not tsl2.is_dense()


def test_tsl_largest_common_contiguous_block():
    tiledStride1 = TiledStride(
        [
            Stride(4, 4),
            Stride(16, 2),
        ]
    )
    tiledStride2 = TiledStride(
        [
            Stride(1, 4),
            Stride(32, 2),
        ]
    )
    tsl1 = TiledStridedLayout([tiledStride1, tiledStride2])
    tiledStride1 = TiledStride(
        [
            Stride(4, 4),
            Stride(32, 2),
        ]
    )
    tiledStride2 = TiledStride(
        [
            Stride(1, 4),
            Stride(16, 2),
        ]
    )
    tsl2 = TiledStridedLayout([tiledStride1, tiledStride2])
    tiledStride1 = TiledStride(
        [
            Stride(8, 4),
            Stride(32, 2),
        ]
    )
    tiledStride2 = TiledStride(
        [
            Stride(1, 4),
            Stride(4, 2),
        ]
    )
    tsl3 = TiledStridedLayout([tiledStride1, tiledStride2])

    ## Check largest common contiguous block between tsl1 and tsl2
    lccb1 = tsl1.largest_common_contiguous_block(tsl2)
    assert len(lccb1) == 2
    assert lccb1[0].stride == 1
    assert lccb1[0].bound == 4
    assert lccb1[1].stride == 4
    assert lccb1[1].bound == 4

    ## other way around should be the same
    lccb2 = tsl2.largest_common_contiguous_block(tsl1)
    assert len(lccb2) == 2
    assert lccb2[0].stride == 1
    assert lccb2[0].bound == 4
    assert lccb2[1].stride == 4
    assert lccb2[1].bound == 4

    lccb3 = tsl1.largest_common_contiguous_block(tsl3)
    assert len(lccb3) == 1
    assert lccb3[0].stride == 1
    assert lccb3[0].bound == 4
