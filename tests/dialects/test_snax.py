# import pytest

from compiler.dialects.snax import TiledStridedLayoutAttr
from xdsl.dialects.builtin import ArrayAttr, IntAttr


def test_tiled_strided_layout():
    tslAttr = TiledStridedLayoutAttr([[1, 2], [3, 4]], [[2, 4], [2, 4]], 0)
    assert tslAttr.strides == ArrayAttr(
        [ArrayAttr([IntAttr(1), IntAttr(2)]), ArrayAttr([IntAttr(3), IntAttr(4)])]
    )
    assert tslAttr.bounds == ArrayAttr(
        [ArrayAttr([IntAttr(2), IntAttr(4)]), ArrayAttr([IntAttr(2), IntAttr(4)])]
    )
    assert tslAttr.offset == IntAttr(0)


def test_tiled_strided_layout_to_tsl():
    strides = [[1, 2], [3, 4]]
    bounds = [[2, 4], [2, 4]]
    tslAttr = TiledStridedLayoutAttr(strides, bounds, 0)
    tslAttr.tsl()
    pass
