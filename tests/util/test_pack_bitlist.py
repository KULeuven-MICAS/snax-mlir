from io import StringIO

from xdsl.dialects import arith, builtin
from xdsl.printer import Printer

from compiler.util.pack_bitlist import pack_bitlist


def test_pack_bitlist():
    out = StringIO()
    p = Printer(stream=out)
    for op in pack_bitlist(
        [1, 128, 2, 128],
        [0, 8, 16, 24],
    ):
        p.print(op)

    assert (
        out.getvalue()
        == """%0 = arith.constant 0 : i32
%1 = arith.constant 1 : i32
%2 = arith.shli %1, %0 : i32
%3 = arith.constant 8 : i32
%4 = arith.constant 128 : i32
%5 = arith.shli %4, %3 : i32
%6 = arith.constant 16 : i32
%7 = arith.constant 2 : i32
%8 = arith.shli %7, %6 : i32
%9 = arith.constant 24 : i32
%10 = arith.constant 128 : i32
%11 = arith.shli %10, %9 : i32
%12 = arith.ori %2, %5 : i32
%13 = arith.ori %8, %11 : i32
%14 = arith.ori %12, %13 : i32
"""
    )


def test_pack_bitlist_mixed_vals():
    out = StringIO()
    p = Printer(stream=out)
    input_vals = [
        arith.Constant.from_int_and_width(v, builtin.i32) for v in [1, 128, 16]
    ]
    for op in input_vals:
        p.print(op)
    for op in pack_bitlist(
        [input_vals[0], input_vals[1], 2, input_vals[1]],
        [0, 8, input_vals[2], 24],
    ):
        p.print(op)

    assert (
        out.getvalue()
        == """%0 = arith.constant 1 : i32
%1 = arith.constant 128 : i32
%2 = arith.constant 16 : i32
%3 = arith.constant 0 : i32
%4 = arith.shli %0, %3 : i32
%5 = arith.constant 8 : i32
%6 = arith.shli %1, %5 : i32
%7 = arith.constant 2 : i32
%8 = arith.shli %7, %2 : i32
%9 = arith.constant 24 : i32
%10 = arith.shli %1, %9 : i32
%11 = arith.ori %4, %6 : i32
%12 = arith.ori %8, %10 : i32
%13 = arith.ori %11, %12 : i32
"""
    )
