import pytest

from compiler.dialects.tsl import TiledStridedLayoutAttr
from compiler.ir.tsl.stride import Stride
from compiler.ir.tsl.tiled_stride import TiledStride
from compiler.ir.tsl.tiled_strided_layout import TiledStridedLayout


@pytest.fixture()
def example_tsl_attr():
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
    tsl = TiledStridedLayout([tiledStride1, tiledStride2])
    tsl_attr = TiledStridedLayoutAttr(tsl)
    return tsl_attr


def test_tsl_attr_constructor(example_tsl_attr):
    tsl = example_tsl_attr
    assert isinstance(tsl, TiledStridedLayoutAttr)
    assert isinstance(tsl.data, TiledStridedLayout)
