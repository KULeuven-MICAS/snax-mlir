# import pytest

# from xdsl.dialects.builtin import ArrayAttr, IntAttr
from compiler.dialects.snax import TiledStridedLayoutAttr

# from compiler.dialects.snax import MemRefType
from xdsl.dialects.memref import MemRefType
from xdsl.dialects.builtin import i32
from xdsl.dialects.builtin import ArrayAttr, IntAttr, IntegerAttr, AnyIntegerAttr

# from xdsl.dialects.memref import *


def test_tiled_strided_layout():
    tslAttr = TiledStridedLayoutAttr([[1, 2], [3, 4]], [[2, 4], [2, 4]], 0)
    assert tslAttr.strides == ArrayAttr(
        [ArrayAttr([IntAttr(1), IntAttr(2)]), ArrayAttr([IntAttr(3), IntAttr(4)])]
    )
    assert tslAttr.bounds == ArrayAttr(
        [ArrayAttr([IntAttr(2), IntAttr(4)]), ArrayAttr([IntAttr(2), IntAttr(4)])]
    )
    assert tslAttr.offset == IntAttr(0)

    shape = ArrayAttr[AnyIntegerAttr](
        [
            d if isinstance(d, IntegerAttr) else IntegerAttr.from_index_int_value(d)
            for d in [2, 4]
        ]
    )

    MemRefType([shape, i32, tslAttr, None])

    # memrefAlloc = Alloc.get(i32, 64, [2, 4], None, tslAttr)
    # prog = ModuleOp([memrefAlloc])
    # printer = Printer(print_generic_format=True)
    # printer.print_op(prog)

    pass


def test_tiled_strided_layout_to_tsl():
    strides = [[1, 2], [3, 4]]
    bounds = [[2, 4], [2, 4]]
    tslAttr = TiledStridedLayoutAttr(strides, bounds, 0)
    tslAttr.tsl()
    pass
