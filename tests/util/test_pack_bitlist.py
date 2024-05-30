from io import StringIO

from compiler.util.pack_bitlist import pack_bitlist
from xdsl.printer import Printer

def test_pack_bitlist():
    out = StringIO()
    p = Printer(stream=out)
    for op in pack_bitlist(
        [1, 128, 2, 128],
        [0, 8, 16, 24],
    ):
        p.print(op)

    assert out.getvalue() == """%0 = arith.constant 0 : i32
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
%13 = arith.ori %12, %8 : i32
%14 = arith.ori %13, %11 : i32
"""