import pytest
from xdsl.ir.affine import AffineMap

from snaxc.dialects.tsl import TiledStridedLayoutAttr
from snaxc.ir.tsl.stride import Stride
from snaxc.ir.tsl.tiled_stride import TiledStride
from snaxc.ir.tsl.tiled_strided_layout import TiledStridedLayout
from snaxc.util.canonicalize_affine import canonicalize_map


@pytest.fixture
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


def test_tsl_attr_constructor(example_tsl_attr: TiledStridedLayoutAttr):
    tsl = example_tsl_attr
    assert isinstance(tsl, TiledStridedLayoutAttr)
    assert isinstance(tsl.data, TiledStridedLayout)


def test_tsl_attr_get_affine(example_tsl_attr: TiledStridedLayoutAttr):
    tsl = example_tsl_attr
    map = canonicalize_map(tsl.get_affine_map())
    assert map == canonicalize_map(
        AffineMap.from_callable(lambda d0, d1: ((((((d0 // 4) * 32) + ((d0 % 4) * 4)) + ((d1 // 4) * 16)) + (d1 % 4)),))
    )
