import pytest

from compiler.util.TiledStridedLayout import Stride, TiledStride, TiledStridedLayout


@pytest.fixture
def example_tiled_strided_layouts():
    t1 = TiledStride(block=Stride(32, 2), tile=Stride(8, 4))
    t2 = TiledStride(block=Stride(4, 2), tile=Stride(1, 4))
    tsl1 = TiledStridedLayout([t1, t2])

    t3 = TiledStride(block=Stride(16, 2), tile=Stride(4, 4))
    t4 = TiledStride(block=Stride(32, 2), tile=Stride(1, 4))
    tsl2 = TiledStridedLayout([t3, t4])

    return tsl1, tsl2


def test_stride_all_values():
    stride = Stride(5, 3)
    assert stride.all_values() == [0, 5, 10]
    stride = Stride(7, 8)
    assert stride.all_values() == [0, 7, 14, 21, 28, 35, 42, 49]


def test_tiled_stride_get_stride():
    block = Stride(32, 2)
    tile = Stride(8, 4)
    tiled_stride = TiledStride(block, tile)

    assert tiled_stride.get_stride(0) == tile
    assert tiled_stride.get_stride(1) == block


def test_tiled_strided_layout_str(example_tiled_strided_layouts):
    tsl1, tsl2 = example_tiled_strided_layouts

    assert str(tsl1) == "([8, 32] x [4, 2], [1, 4] x [4, 2])"
    assert str(tsl2) == "([4, 16] x [4, 2], [1, 32] x [4, 2])"


def test_tiled_strided_layout_iteration(example_tiled_strided_layouts):
    tsl1, _ = example_tiled_strided_layouts
    strides = list(iter(tsl1))

    assert len(strides) == 4
    assert isinstance(strides[0], Stride)
    assert isinstance(strides[1], Stride)
    assert isinstance(strides[2], Stride)
    assert isinstance(strides[3], Stride)


def test_tiled_strided_layout_is_dense(example_tiled_strided_layouts):
    tsl1, _ = example_tiled_strided_layouts

    assert tsl1.is_dense()


def test_tiled_strided_layout_self_overlaps(example_tiled_strided_layouts):
    tsl1, _ = example_tiled_strided_layouts

    assert not tsl1.self_overlaps()


def test_tiled_strided_layout_get_all_values(example_tiled_strided_layouts):
    tsl1, _ = example_tiled_strided_layouts

    assert set(tsl1.get_all_values()) == set(range(0, 64))


def test_tiled_strided_layout_largest_common_contiguous_block(
    example_tiled_strided_layouts,
):
    tsl1, tsl2 = example_tiled_strided_layouts

    result = tsl1.largest_common_contiguous_block(tsl2)
    assert len(result) == 1
    assert isinstance(result[0], Stride)
    assert result[0].stride == 1


def test_tiled_strided_layout_get_2d_dma_strategy(
    example_tiled_strided_layouts, capsys
):
    tsl1, tsl2 = example_tiled_strided_layouts

    tsl1.get_2d_dma_strategy(tsl2)

    # captured = capsys.readouterr()
    # expected_output = (
    #     "\n"
    #     "for(int i = 0; i < 4; i++)\n"
    #     "    snrt_dma_start_2d(*dst + i * 1, *src + i * 8,
    # size=32, dst_stride=32, src_stride=1, repeat=2);\n"
    #     "for(int i = 0; i < 2; i++)\n"
    #     "    snrt_dma_start_2d(*dst + i * 32, *src + i * 4, size=32,
    # dst_stride=1, src_stride=16, repeat=2);\n\n"
    # )

    # assert captured.out == expected_output


if __name__ == "__main__":
    pytest.main()
