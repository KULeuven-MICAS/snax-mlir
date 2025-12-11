import math
from typing import cast

from xdsl import printer
from xdsl.dialects.builtin import i32
from xdsl.dialects.hw import ArrayType
from xdsl.dialects.test import TestOp
from xdsl.ir import Block, SSAValue, StringIO

from snaxc.phs.hw_conversion import create_shaped_hw_array, create_shaped_hw_array_type, get_from_shaped_hw_array


def test_create_shaped_hw_array_type():
    assert str(create_shaped_hw_array_type(i32, (1, 2, 3))) == "!hw.array<1x!hw.array<2x!hw.array<3xi32>>>"
    assert str(create_shaped_hw_array_type(i32, ())) == "i32"  # 0D edge case
    assert str(create_shaped_hw_array_type(i32, (3, 3, 3))) == "!hw.array<3x!hw.array<3x!hw.array<3xi32>>>"
    assert str(create_shaped_hw_array_type(i32, (4,))) == "!hw.array<4xi32>"


def test_get_from_shaped_hw_array():
    block = Block(arg_types=(create_shaped_hw_array_type(i32, (2, 1, 3)),))
    operations, result = get_from_shaped_hw_array(cast(SSAValue[ArrayType], SSAValue.get(block.args[0])), (1, 1, 3))
    block.add_ops([*operations, TestOp([result])])
    stream = StringIO()
    printr = printer.Printer(stream=stream)
    printr.print_block(block)
    expected = """\
\n^bb0(%0 : !hw.array<2x!hw.array<1x!hw.array<3xi32>>>):\
\n  %1 = arith.constant -1 : i2\
\n  %2 = hw.array_get %0[%1] : !hw.array<2x!hw.array<1x!hw.array<3xi32>>>, i2\
\n  %3 = arith.constant 1 : i2\
\n  %4 = hw.array_get %2[%3] : !hw.array<1x!hw.array<3xi32>>, i2\
\n  %5 = arith.constant 1 : i2\
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
\n  %6 = hw.array_create %0, %1 : i32\
\n  %7 = hw.array_create %2, %3 : i32\
\n  %8 = hw.array_create %4, %5 : i32\
\n  %9 = hw.array_create %6, %7, %8 : !hw.array<2xi32>\
\n  "test.op"(%9) : (!hw.array<3x!hw.array<2xi32>>) -> ()\
"""
    gotten = stream.getvalue()
    assert gotten == expected


if __name__ == "__main__":
    test_create_shaped_hw_array_type()
    test_get_from_shaped_hw_array()
    test_create_shaped_hw_array()
