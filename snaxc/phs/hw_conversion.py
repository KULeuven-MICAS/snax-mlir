import math
from typing import cast

from xdsl.dialects import arith, builtin, hw
from xdsl.ir import Attribute, BlockArgument, Operation, SSAValue, TypeAttribute
from xdsl.utils.hints import isa

from snaxc.dialects import phs


def get_shaped_hw_array_shape(array_type: hw.ArrayType) -> tuple[list[int], Attribute]:
    el_type = array_type.get_element_type()
    this_shape = array_type.size_attr.data
    if not isa(el_type, hw.ArrayType):
        return [this_shape], el_type
    else:
        sub_shape, el_type = get_shaped_hw_array_shape(el_type)
        return [this_shape, *sub_shape], el_type


def create_shaped_hw_array_type(
    el_type: builtin.AnySignlessIntegerType | hw.ArrayType, shape: tuple[int, ...]
) -> builtin.AnySignlessIntegerType | hw.ArrayType:
    """
    Generates a recursively nested !hw.array type based on a given shape, e.g.:

    i32, (1,2,3) -> !hw.array<1x!hw.array<2x!hw.array<3xi32>>>
    i32, (2) -> !hw.array<2xi32>
    i32, () -> i32
    """

    # Edge case for "0D" arrays
    if len(shape) == 0:
        return el_type

    if len(shape) == 1:
        return hw.ArrayType(el_type, shape[0])

    else:
        return hw.ArrayType(create_shaped_hw_array_type(el_type, shape[1:]), shape[0])


const_or_array_list = list[arith.ConstantOp | hw.ArrayGetOp]


def get_from_shaped_hw_array(
    input_val: SSAValue[hw.ArrayType], index: tuple[int, ...]
) -> tuple[const_or_array_list, SSAValue]:
    """
    Generate array_get ops with arith.constants indexes to get a value out of a nested !hw.array e.g.:

    input: An operation that outputs !hw.array<2x!hw.array<3xi32>>
    index: (1,2)

    Outputs:
        %cst_1 = arith.constant : 1 i1
        %first_get = hw.array_get %input[%cst_1]
        %cst_2 = arith.constant : 2 i2
        %return_out = hw.array_get %first_get[%cst_2]

    The result value from return_out is given as the second return value
    """
    assert len(index) > 0, "Expect index to have at least one value"

    # Perform static type check
    shape, _ = get_shaped_hw_array_shape(input_val.type)
    for i, size in zip(index, shape, strict=True):
        if not i < size:
            raise ValueError(f"Size {i} will be out of bounds for a size of {size}")

    # If sizes are okay, continue
    def get_from_shaped_hw_array_inner(
        input_val: SSAValue[hw.ArrayType], index: tuple[int, ...]
    ) -> tuple[const_or_array_list, SSAValue]:
        bitwidth = (input_val.type.size_attr.data - 1).bit_length()
        index_op = arith.ConstantOp.from_int_and_width(index[0], bitwidth)
        array_get_op = hw.ArrayGetOp(input_val, index_op)
        ops: const_or_array_list = [index_op, array_get_op]
        if len(index) == 1:
            return ops, array_get_op.result
        else:
            sub_ops, final_result = get_from_shaped_hw_array_inner(
                SSAValue.get(array_get_op, type=hw.ArrayType), index[1:]
            )
            ops.extend(sub_ops)
            return ops, final_result

    return get_from_shaped_hw_array_inner(input_val, index)


def create_shaped_hw_array(
    values: list[SSAValue[Attribute]], shape: tuple[int, ...]
) -> tuple[list[Operation], SSAValue]:
    """
    Returns multiple array_create operations until a nested array is formed

    %6 = hw.array_create %0, %1 : i32
    %7 = hw.array_create %2, %3 : i32
    %8 = hw.array_create %4, %5 : i32
    %9 = hw.array_create %6, %7, %8 : !hw.array<2xi32>
    "test.op"(%9) : (!hw.array<3x!hw.array<2xi32>>) -> ()
    """
    length = len(values)
    expected = math.prod(shape)
    assert length == expected, f"Values list length {length} doesn't match {shape} (expected {expected})"

    def create_shaped_hw_array_inner(
        values: list[SSAValue[Attribute]], shape: tuple[int, ...]
    ) -> tuple[list[Operation], SSAValue]:
        if len(shape) == 1:
            op = hw.ArrayCreateOp(*values)
            return [op], op.result
        else:
            ops: list[Operation] = []
            new_vals: list[SSAValue] = []
            sub_array_size = math.prod(shape[1:])
            for i in range(shape[0]):
                start_idx = i * sub_array_size
                end_idx = start_idx + sub_array_size
                sub_values = values[start_idx:end_idx]
                new_op, new_result = create_shaped_hw_array_inner(sub_values, shape[1:])
                ops.extend(new_op)
                new_vals.append(new_result)
            op = hw.ArrayCreateOp(*new_vals)
            ops.append(op)
            return ops, op.result

    return create_shaped_hw_array_inner(values, shape)


def get_choice_bitwidth(choice: phs.ChooseOp) -> int:
    """
    Get the amount of bits necessary to represent the choices in a ChooseOp.
    i.e. ceil(log2(n)) bits for n choices.
    """
    choices = len(list(choice.operations()))
    assert choices > 0, "Expect choose_op to have at least one choice"
    return (choices - 1).bit_length()


def get_switch_bitwidth(arg: BlockArgument) -> int:
    """
    Get the amount of bits necessary to represent the choices made by a certain switch:
    * returns 1 if the user is a phs.MuxOp
    * returns ceil(log2(n)) if the user is a phs.ChooseOp with n choices
    Throws an assertion error if the switch has multiple users
    """
    use = arg.get_unique_use()
    assert use is not None, "Expect single user for switch"
    if isinstance(use.operation, phs.ChooseOp):
        return get_choice_bitwidth(use.operation)
    elif isinstance(use.operation, phs.MuxOp):
        return 1
    else:
        raise NotImplementedError(f"got {use}")


def get_pe_port_decl(pe: phs.PEOp) -> builtin.ArrayAttr[hw.ModulePort]:
    ports: list[hw.ModulePort] = []
    for i, data_opnd in enumerate(pe.data_operands()):
        ports.append(
            hw.ModulePort(
                builtin.StringAttr(f"data_{i}"),
                cast(TypeAttribute, data_opnd.type),
                hw.DirectionAttr(data=hw.Direction.INPUT),
            )
        )
    for i, switch in enumerate(pe.get_switches()):
        ports.append(
            hw.ModulePort(
                builtin.StringAttr(f"switch_{i}"),
                builtin.IntegerType(get_switch_bitwidth(switch)),
                hw.DirectionAttr(data=hw.Direction.INPUT),
            )
        )
    for i, output in enumerate(pe.get_terminator().operands):
        ports.append(
            hw.ModulePort(
                builtin.StringAttr(f"out_{i}"),
                cast(TypeAttribute, output.type),
                hw.DirectionAttr(data=hw.Direction.OUTPUT),
            )
        )
    return builtin.ArrayAttr(ports)
