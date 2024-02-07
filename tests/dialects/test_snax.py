import pytest
from xdsl.dialects import builtin
from xdsl.dialects.builtin import StridedLayoutAttr, i32, i64
from xdsl.dialects.memref import MemRefType
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import TestSSAValue

from compiler.dialects.snax import ReShuffleOp


def test_memref_memory_space_cast():
    layout_1 = StridedLayoutAttr(strides=(2, 4, 6), offset=8)
    layout_2 = StridedLayoutAttr(strides=(2, 8, 16), offset=4)

    source_type = MemRefType(
        i32, [10, 2], layout=layout_1, memory_space=builtin.IntegerAttr(1, i32)
    )
    source_ssa = TestSSAValue(source_type)

    dest_type = MemRefType(i32, [10, 2], memory_space=builtin.IntegerAttr(1, i32))

    reshuffle_op = ReShuffleOp(source_ssa, dest_type)

    assert reshuffle_op.source is source_ssa
    assert reshuffle_op.dest.type is dest_type

    dest_type_other_element = MemRefType(
        i64, [10, 2], layout=layout_2, memory_space=builtin.IntegerAttr(1, i32)
    )

    with pytest.raises(
        VerifyException,
        match="Expected source and destination to have the same element type.",
    ):
        ReShuffleOp(source_ssa, dest_type_other_element).verify()

    dest_type_other_shape = MemRefType(
        i32, [10, 4], layout=layout_2, memory_space=builtin.IntegerAttr(1, i32)
    )

    with pytest.raises(
        VerifyException, match="Expected source and destination to have the same shape."
    ):
        ReShuffleOp(source_ssa, dest_type_other_shape).verify()

    dest_type_other_space = MemRefType(
        i32, [10, 2], layout=layout_2, memory_space=builtin.IntegerAttr(2, i32)
    )

    with pytest.raises(
        VerifyException,
        match="Expected source and destination to have the same memory space.",
    ):
        ReShuffleOp(source_ssa, dest_type_other_space).verify()

    type_nolayout = MemRefType(i32, [10, 2], memory_space=builtin.IntegerAttr(1, i32))
    TestSSAValue(type_nolayout)

    # Test helper function
    reshuffle_op = ReShuffleOp.from_type_and_target_layout(source_ssa, layout_2)

    assert reshuffle_op.source is source_ssa
    assert isinstance(reshuffle_op.dest.type, MemRefType)
    assert reshuffle_op.dest.type.layout is layout_2
