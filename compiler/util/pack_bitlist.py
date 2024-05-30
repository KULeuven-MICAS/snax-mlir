from typing import Sequence, Iterable
from xdsl.ir import Operation, SSAValue
from xdsl.dialects import arith


def pack_bitlist(values: Sequence[int | SSAValue | Operation], offsets: Sequence[int | SSAValue | Operation], dtype : int = 32) -> Iterable[Operation]:
    """
    Takes a list of values and offsets, and packs them into a single `dtype` wide value.

    Each value is shifted left by its offset and then all of them are or-ed together.
    (They are or-ed together in a way that reduces data dependencies and makes it O(log(n)))

    Values and offsets can be passed as either SSA values or integers.
    """
    shifted_vals: list[SSAValue] = []
    for int_val, int_off in zip(values, offsets, strict=True):
        if isinstance(int_off, int):
            yield (offset := arith.Constant.from_int_and_width(int_off, dtype))
        else:
            offset = SSAValue.get(int_off)
        if isinstance(int_val, int):
            yield (value := arith.Constant.from_int_and_width(int_val, dtype))
        else:
            value = SSAValue.get(int_val)
        yield (shift := arith.ShLI(value, offset))
        shifted_vals.append(shift.result)

    # do a divide-and-conquer algorithm to or the values together in O(log(n)) instead of O(n).
    while len(shifted_vals) > 1:
        a, b, *rest = shifted_vals
        yield (ored_val := arith.OrI(a, b))
        shifted_vals = [*rest, ored_val.result]
