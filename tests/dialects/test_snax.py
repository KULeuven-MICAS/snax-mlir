import pytest
from xdsl.dialects import builtin
from xdsl.dialects.builtin import StridedLayoutAttr, i32, i64
from xdsl.dialects.llvm import LLVMPointerType
from xdsl.dialects.memref import MemRefType
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import TestSSAValue

from compiler.dialects.snax import Alloc, LayoutCast


def test_memref_memory_space_cast():
    layout_1 = StridedLayoutAttr(strides=(2, 4, 6), offset=8)
    layout_2 = StridedLayoutAttr(strides=(2, 8, 16), offset=4)

    source_type = MemRefType(
        i32, [10, 2], layout=layout_1, memory_space=builtin.IntegerAttr(1, i32)
    )
    source_ssa = TestSSAValue(source_type)

    dest_type = MemRefType(i32, [10, 2], memory_space=builtin.IntegerAttr(1, i32))

    memory_layout_cast = LayoutCast(source_ssa, dest_type)

    assert memory_layout_cast.source is source_ssa
    assert memory_layout_cast.dest.type is dest_type

    dest_type_other_element = MemRefType(
        i64, [10, 2], layout=layout_2, memory_space=builtin.IntegerAttr(1, i32)
    )

    with pytest.raises(
        VerifyException,
        match="Expected source and destination to have the same element type.",
    ):
        LayoutCast(source_ssa, dest_type_other_element).verify()

    dest_type_other_shape = MemRefType(
        i32, [10, 4], layout=layout_2, memory_space=builtin.IntegerAttr(1, i32)
    )

    with pytest.raises(
        VerifyException, match="Expected source and destination to have the same shape."
    ):
        LayoutCast(source_ssa, dest_type_other_shape).verify()

    dest_type_other_space = MemRefType(
        i32, [10, 2], layout=layout_2, memory_space=builtin.IntegerAttr(2, i32)
    )

    with pytest.raises(
        VerifyException,
        match="Expected source and destination to have the same memory space.",
    ):
        LayoutCast(source_ssa, dest_type_other_space).verify()

    type_nolayout = MemRefType(i32, [10, 2], memory_space=builtin.IntegerAttr(1, i32))
    TestSSAValue(type_nolayout)

    # Test helper function
    memory_layout_cast = LayoutCast.from_type_and_target_layout(source_ssa, layout_2)

    assert memory_layout_cast.source is source_ssa
    assert isinstance(memory_layout_cast.dest.type, MemRefType)
    assert memory_layout_cast.dest.type.layout is layout_2


def test_snax_alloc():
    size = TestSSAValue(i32)
    alloc_a = Alloc(size, memory_space=builtin.IntegerAttr(1, i32))

    assert alloc_a.size is size
    assert alloc_a.memory_space.value.data == 1
    assert isinstance(alloc_a.result.type, LLVMPointerType)
