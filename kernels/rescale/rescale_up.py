import os
from collections.abc import Sequence
from io import StringIO

import numpy as np
from xdsl.builder import Builder
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    BoolAttr,
    DenseIntOrFPElementsAttr,
    ModuleOp,
    StringAttr,
    TensorType,
    i1,
    i8,
    i32,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.tosa import ConstOp, RescaleOp
from xdsl.printer import Printer


def rescale_up(n: int = 64):
    np.random.seed(42)  # For reproducibility
    # Define Variables For Program:
    a_type = TensorType(i8, (n,))
    # a_vals = np.full((n,), -8737248 , dtype=np.int32)
    a_vals = np.random.randint(  # pyright: ignore[reportUnknownMemberType]
        -127,
        128,
        n,
    )
    # print(a_vals)

    shift = 10
    multiplier = 10283821
    input_zp = 0
    output_zp = 0
    max_int = 2147483647
    min_int = -2147483648

    output_type = TensorType(i32, (n,))
    golden_vals_list: Sequence[int] = []
    for val in a_vals:
        golden_vals_list.append(golden_model_rescale_up(val, input_zp, output_zp, shift, max_int, min_int, multiplier))

    golden_vals = np.array(golden_vals_list, dtype=np.int32)

    res_types = [output_type] * 2

    # Define Program:

    @Builder.implicit_region([])
    def func_body(_) -> None:
        # Declare constants
        a = ConstantOp(DenseIntOrFPElementsAttr.from_list(a_type, a_vals.flatten().tolist()))
        print(a_vals.flatten().tolist())
        golden = ConstantOp(DenseIntOrFPElementsAttr.from_list(output_type, golden_vals.flatten().tolist()))

        # # Declare result tensor type
        # empty_tensor = EmptyOp([], output_type)

        # Specify the operation
        input_zp_op = ConstOp(DenseIntOrFPElementsAttr.from_list(TensorType(i32, (1,)), (input_zp,)))
        output_zp_op = ConstOp(DenseIntOrFPElementsAttr.from_list(TensorType(i32, (1,)), (output_zp,)))
        multiplier_op = ConstOp(DenseIntOrFPElementsAttr.from_list(TensorType(i32, (1,)), (multiplier,)))
        shift_op = ConstOp(DenseIntOrFPElementsAttr.from_list(TensorType(i8, (1,)), (shift,)))
        result = RescaleOp(
            operands=[a.result, multiplier_op, shift_op, input_zp_op, output_zp_op],
            result_types=[output_type],
            properties={
                "scale32": BoolAttr(True, i1),
                "rounding_mode": StringAttr("DOUBLE_ROUND"),
                "per_channel": BoolAttr(False, i1),
                "input_unsigned": BoolAttr(False, i1),
                "output_unsigned": BoolAttr(False, i1),
            },
        )

        # Return both the computed result and the golden output
        ReturnOp(result, golden)

    function = FuncOp.from_region("snax_main", [], res_types, func_body)
    return ModuleOp([function])


def golden_model_rescale_up(
    data_in: int,
    input_zp_i: int,
    output_zp_i: int,
    shift_i: int,
    max_int_i: int,
    min_int_i: int,
    multiplier_i: int,
) -> int:
    """
    This function performs rescaling of data given exact algorithm of TOSA.rescale,
    """
    # Step 1: Subtract input zero point
    var_1 = data_in - input_zp_i

    # Step 2: Multiply with the multiplier avoiding overflow
    var_2 = np.int64(var_1) * np.int64(multiplier_i)

    # Step 3: Left shift one
    shifted_one = np.int64(1 << (shift_i - 1))  # TODO: check if the minus one is actually correct

    # Step 4: Add shifted one
    var_3 = np.int64(var_2 + shifted_one)

    # Step 6: Shift right
    var_6 = np.int32(var_3 >> shift_i)

    # Step 7: Add output zero point
    var_7 = var_6 + np.int32(output_zp_i)

    # Step 8: Clip the values to be within min and max integer range
    var_8 = np.clip(var_7, min_int_i, max_int_i)

    return int(var_8)


if __name__ == "__main__":
    # Get the name of the current Python script and replace its extension with .mlir
    script_name = os.path.basename(__file__)
    mlir_filename = os.path.splitext(script_name)[0] + ".ppp.mlir"

    # Generate IR and write it to the specified MLIR file
    output = StringIO()
    printer = Printer(stream=output)
    printer.print(rescale_up())
    with open(mlir_filename, "w") as output_file:
        output_file.write(output.getvalue())
