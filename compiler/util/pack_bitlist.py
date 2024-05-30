from typing import Sequence, Iterable
from xdsl.ir import Operation, Attribute
from xdsl.dialects import arith, builtin


def pack_bitlist(values: Sequence[int], offsets: Sequence[int], dtype : Attribute = builtin.i32) -> Iterable[Operation]:
    shifted_vals = []
    for int_val, int_off in zip(values, offsets, strict=True):
        yield (offset := arith.Constant.from_int_and_width(int_off, dtype))
        yield (value := arith.Constant.from_int_and_width(int_val, dtype))
        yield (shift := arith.ShLI(value, offset))
        shifted_vals.append(shift)

    carry = shifted_vals[0]
    for val in shifted_vals[1:]:
        yield (carry := arith.OrI(carry, val))
