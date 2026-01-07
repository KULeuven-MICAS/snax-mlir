import math
from typing import cast

from xdsl import printer
from xdsl.dialects.arith import AddfOp, DivfOp, MaximumfOp, MulfOp, SubfOp
from xdsl.dialects.builtin import Float32Type, IndexType, i32
from xdsl.dialects.hw import ArrayType
from xdsl.dialects.test import TestOp
from xdsl.ir import Attribute, Block, SSAValue, StringIO

from snaxc.dialects import phs
from snaxc.phs.hw_conversion import (
    create_shaped_hw_array,
    create_shaped_hw_array_type,
    get_choice_bitwidth,
    get_from_shaped_hw_array,
    get_shaped_hw_array_shape,
    get_switch_bitwidth,
)


def test_create_shaped_hw_array_type():
    assert str(create_shaped_hw_array_type(i32, (1, 2, 3))) == "!hw.array<1x!hw.array<2x!hw.array<3xi32>>>"
    assert str(create_shaped_hw_array_type(i32, ())) == "i32"  # 0D edge case
    assert str(create_shaped_hw_array_type(i32, (3, 3, 3))) == "!hw.array<3x!hw.array<3x!hw.array<3xi32>>>"
    assert str(create_shaped_hw_array_type(i32, (4,))) == "!hw.array<4xi32>"


def test_get_from_shaped_hw_array():
    block = Block(arg_types=(create_shaped_hw_array_type(i32, (2, 1, 3)),))
    operations, result = get_from_shaped_hw_array(SSAValue.get(block.args[0], type=ArrayType), (1, 0, 2))
    block.add_ops([*operations, TestOp([result])])
    stream = StringIO()
    printr = printer.Printer(stream=stream)
    printr.print_block(block)
    expected = """\
\n^bb0(%0 : !hw.array<2x!hw.array<1x!hw.array<3xi32>>>):\
\n  %1 = arith.constant true\
\n  %2 = hw.array_get %0[%1] : !hw.array<2x!hw.array<1x!hw.array<3xi32>>>, i1\
\n  %3 = arith.constant 0 : i0\
\n  %4 = hw.array_get %2[%3] : !hw.array<1x!hw.array<3xi32>>, i0\
\n  %5 = arith.constant -2 : i2\
\n  %6 = hw.array_get %4[%5] : !hw.array<3xi32>, i2\
\n  "test.op"(%6) : (i32) -> ()"""
    gotten = stream.getvalue()
    assert gotten == expected


def test_create_shaped_hw_array():
    shape = (3, 2)
    test_ops = [TestOp(result_types=[i32]) for _ in range(math.prod(shape))]
    vals = [SSAValue.get(op.results[0]) for op in test_ops]
    operations, result = create_shaped_hw_array(vals, shape)
    test_op = TestOp([result])
    stream = StringIO()
    printr = printer.Printer(stream=stream)
    printr.print_block(Block([*test_ops, *operations, test_op]))
    expected = """\
\n^bb0:\
\n  %0 = "test.op"() : () -> i32\
\n  %1 = "test.op"() : () -> i32\
\n  %2 = "test.op"() : () -> i32\
\n  %3 = "test.op"() : () -> i32\
\n  %4 = "test.op"() : () -> i32\
\n  %5 = "test.op"() : () -> i32\
\n  %6 = hw.array_create %1, %0 : i32\
\n  %7 = hw.array_create %3, %2 : i32\
\n  %8 = hw.array_create %5, %4 : i32\
\n  %9 = hw.array_create %8, %7, %6 : !hw.array<2xi32>\
\n  "test.op"(%9) : (!hw.array<3x!hw.array<2xi32>>) -> ()\
"""
    gotten = stream.getvalue()
    assert gotten == expected


def test_get_shaped_hw_array_shape():
    input_type = i32
    input_shape = (1, 2, 3)
    array_type = create_shaped_hw_array_type(input_type, input_shape)
    array_type = cast(ArrayType[Attribute], array_type)
    output_shape, output_type = get_shaped_hw_array_shape(array_type)
    assert input_shape == tuple(output_shape)
    assert input_type == output_type


def test_get_bitwidth():
    switch_types = [IndexType(), IndexType(), IndexType()]
    # Based on operation
    in_types = [Float32Type(), Float32Type()]
    out_types = [Float32Type()]
    # Construct a new block based on the input of the
    block_inputs = [*in_types, *switch_types]
    blockA = Block(arg_types=block_inputs)
    lhs, rhs, switch, switch2, switch3 = blockA.args
    addf_op = AddfOp(lhs, rhs)
    subf_op = SubfOp(lhs, rhs)
    divf_op = DivfOp(lhs, rhs)
    mulf_op = MulfOp(lhs, rhs)
    maxf_op = MaximumfOp(lhs, rhs)
    blockA.add_ops(
        [
            result := phs.ChooseOp.from_operations(
                "_0", [lhs, rhs], switch, [addf_op, subf_op, divf_op, mulf_op, maxf_op], out_types
            ),
            result2 := phs.ChooseOp.from_operations("_0", [lhs, rhs], switch2, [addf_op, subf_op, divf_op], out_types),
            mux := phs.MuxOp(result, result2, switch3),
            phs.YieldOp(mux),
        ]
    )
    assert get_choice_bitwidth(result) == 3
    assert get_choice_bitwidth(result2) == 2
    assert get_switch_bitwidth(switch) == 3
    assert get_switch_bitwidth(switch2) == 2
    assert get_switch_bitwidth(switch3) == 1


if __name__ == "__main__":
    test_create_shaped_hw_array_type()
    test_get_from_shaped_hw_array()
    test_create_shaped_hw_array()
    test_get_shaped_hw_array_shape()
    test_get_bitwidth()
