import pytest

from compiler.dialects.tsl import StrideAttr, TiledStrideAttr, TiledStridedLayoutAttr


@pytest.fixture()
def example_strides():
    return (StrideAttr(1, 4), StrideAttr(4, 6), StrideAttr(24, 2))


def test_stride_constructor(example_strides):
    stride1, stride2, _ = example_strides
    assert stride1.stride.data == 1
    assert stride1.bound.data == 4
    assert stride2.stride.data == 4
    assert stride2.bound.data == 6


def test_stride_all_values(example_strides):
    stride1, stride2, _ = example_strides
    assert stride1.all_values() == [0, 1, 2, 3]
    assert stride2.all_values() == [0, 4, 8, 12, 16, 20]


def test_stride_str(example_strides):
    stride1, stride2, stride3 = example_strides
    assert str(stride1) == "1 x 4"
    assert str(stride2) == "4 x 6"
    assert str(stride3) == "24 x 2"


@pytest.fixture()
def example_tiled_strides(example_strides):
    stride1, stride2, stride3 = example_strides
    tiledStride1 = TiledStrideAttr([stride1, stride2])
    tiledStride2 = TiledStrideAttr([stride1, stride2, stride3])
    return tiledStride1, tiledStride2


def test_tiled_stride_constructor(example_strides, example_tiled_strides):
    stride1, stride2, stride3 = example_strides
    tiledStride1, tiledStride2 = example_tiled_strides
    assert tiledStride1.strides.data[0] == stride1
    assert tiledStride1.strides.data[1] == stride2
    assert tiledStride2.strides.data[0] == stride1
    assert tiledStride2.strides.data[1] == stride2
    assert tiledStride2.strides.data[2] == stride3


def test_tiled_stride_depth(example_tiled_strides):
    tiledStride1, tiledStride2 = example_tiled_strides
    assert tiledStride1.depth() == 2
    assert tiledStride2.depth() == 3


def test_tiled_stride_str(example_tiled_strides):
    tiledStride1, tiledStride2 = example_tiled_strides
    assert str(tiledStride1) == "[1, 4] x [4, 6]"
    assert str(tiledStride2) == "[1, 4, 24] x [4, 6, 2]"


def test_tiled_stride_iter(example_strides, example_tiled_strides):
    strides = example_strides
    _, tiledStride2 = example_tiled_strides

    for depth, stride in tiledStride2:
        assert isinstance(depth, int)
        assert isinstance(stride, StrideAttr)
        assert stride == strides[depth]


@pytest.fixture()
def example_tsl():
    tiledStride1 = TiledStrideAttr(
        [
            StrideAttr(4, 4),
            StrideAttr(32, 2),
        ]
    )
    tiledStride2 = TiledStrideAttr(
        [
            StrideAttr(1, 4),
            StrideAttr(16, 2),
        ]
    )
    tsl = TiledStridedLayoutAttr([tiledStride1, tiledStride2])
    return tsl


def test_tsl_constructor(example_tsl):
    tsl = example_tsl
    assert isinstance(tsl.tstrides.data[0], TiledStrideAttr)
    assert isinstance(tsl.tstrides.data[1], TiledStrideAttr)


def test_tsl_str(example_tsl):
    tsl = example_tsl
    assert str(tsl) == "([4, 32] x [4, 2], [1, 16] x [4, 2])"


def test_tsl_iter(example_tsl):
    tsl = example_tsl
    count = 0
    for dim, depth, stride in tsl:
        count += 1
        assert isinstance(dim, int)
        assert isinstance(depth, int)
        assert isinstance(stride, StrideAttr)
        assert stride == tsl.tstrides.data[dim].strides.data[depth]
    assert count == tsl.dimension() * tsl.tstrides.data[0].depth()


def test_tsl_all_values(example_tsl):
    tsl = example_tsl
    assert set(tsl.all_values()) == set(range(64))


def test_tsl_self_overlaps(example_tsl):
    tsl = example_tsl
    assert not tsl.self_overlaps()

    tiledStride1 = TiledStrideAttr(
        [
            StrideAttr(4, 4),
            StrideAttr(16, 2),
        ]
    )
    tiledStride2 = TiledStrideAttr(
        [
            StrideAttr(1, 4),
            StrideAttr(16, 2),
        ]
    )
    tsl2 = TiledStridedLayoutAttr([tiledStride1, tiledStride2])

    assert tsl2.self_overlaps()


def test_tsl_is_dense(example_tsl):
    tsl = example_tsl
    assert tsl.is_dense()

    tiledStride1 = TiledStrideAttr(
        [
            StrideAttr(4, 4),
            StrideAttr(64, 2),
        ]
    )
    tiledStride2 = TiledStrideAttr(
        [
            StrideAttr(1, 4),
            StrideAttr(16, 2),
        ]
    )
    tsl2 = TiledStridedLayoutAttr([tiledStride1, tiledStride2])

    assert not tsl2.is_dense()


def test_tsl_largest_common_contiguous_block():
    tiledStride1 = TiledStrideAttr(
        [
            StrideAttr(4, 4),
            StrideAttr(16, 2),
        ]
    )
    tiledStride2 = TiledStrideAttr(
        [
            StrideAttr(1, 4),
            StrideAttr(32, 2),
        ]
    )
    tsl1 = TiledStridedLayoutAttr([tiledStride1, tiledStride2])
    tiledStride1 = TiledStrideAttr(
        [
            StrideAttr(4, 4),
            StrideAttr(32, 2),
        ]
    )
    tiledStride2 = TiledStrideAttr(
        [
            StrideAttr(1, 4),
            StrideAttr(16, 2),
        ]
    )
    tsl2 = TiledStridedLayoutAttr([tiledStride1, tiledStride2])
    tiledStride1 = TiledStrideAttr(
        [
            StrideAttr(8, 4),
            StrideAttr(32, 2),
        ]
    )
    tiledStride2 = TiledStrideAttr(
        [
            StrideAttr(1, 4),
            StrideAttr(4, 2),
        ]
    )
    tsl3 = TiledStridedLayoutAttr([tiledStride1, tiledStride2])

    ## Check largest common contiguous block between tsl1 and tsl2
    lccb1 = tsl1.largest_common_contiguous_block(tsl2)
    assert len(lccb1) == 2
    assert lccb1[0].stride.data == 1
    assert lccb1[0].bound.data == 4
    assert lccb1[1].stride.data == 4
    assert lccb1[1].bound.data == 4

    ## other way around should be the same
    lccb2 = tsl2.largest_common_contiguous_block(tsl1)
    assert len(lccb2) == 2
    assert lccb2[0].stride.data == 1
    assert lccb2[0].bound.data == 4
    assert lccb2[1].stride.data == 4
    assert lccb2[1].bound.data == 4

    lccb3 = tsl1.largest_common_contiguous_block(tsl3)
    assert len(lccb3) == 1
    assert lccb3[0].stride.data == 1
    assert lccb3[0].bound.data == 4
