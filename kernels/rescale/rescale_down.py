import os
from collections.abc import Sequence
from io import StringIO

import numpy as np
from xdsl.builder import Builder
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    BoolAttr,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    IntegerAttr,
    ModuleOp,
    TensorType,
    i1,
    i8,
    i32,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.tosa import RescaleOp
from xdsl.printer import Printer


def rescale_down(n: int = 64):
    np.random.seed(42)  # For reproducibility
    # Define Variables For Program:
    a_type = TensorType(i32, (n,))
    # a_vals = np.full((n,), -8737248 , dtype=np.int32)
    a_vals = np.random.randint(
        -10000000,
        10000000,
        n,
    )
    # print(a_vals)

    output_type = TensorType(i8, (n,))
    golden_vals_list: Sequence[int] = []
    for val in a_vals:
        golden_vals_list.append(
            golden_model_rescale_down(val, 0, 0, 47, 127, -128, True, 1140768826)
        )

    golden_vals = np.array(golden_vals_list, dtype=np.int8)

    res_types = [output_type] * 2

    # Define Program:

    @Builder.implicit_region([])
    def func_body(_) -> None:
        # Declare constants
        a = ConstantOp(
            DenseIntOrFPElementsAttr.from_list(a_type, a_vals.flatten().tolist())
        )
        print(a_vals.flatten().tolist())
        golden = ConstantOp(
            DenseIntOrFPElementsAttr.from_list(
                output_type, golden_vals.flatten().tolist()
            )
        )

        # # Declare result tensor type
        # empty_tensor = EmptyOp([], output_type)

        # Specify the operation
        result = RescaleOp(
            operands=[a.result],
            result_types=[output_type],
            properties={
                "input_zp": IntegerAttr(0, i32),
                "output_zp": IntegerAttr(0, i32),
                "multiplier": DenseArrayBase.create_dense_int(i32, [1140768826]),
                "shift": DenseArrayBase.create_dense_int(i32, [47]),
                "scale32": BoolAttr(True, i1),
                "double_round": BoolAttr(True, i1),
                "per_channel": BoolAttr(False, i1),
            },
        )

        # Return both the computed result and the golden output
        ReturnOp(result, golden)

    function = FuncOp.from_region("snax_main", [], res_types, func_body)
    return ModuleOp([function])


def golden_model_rescale_down(
    data_in: int,
    input_zp_i: int,
    output_zp_i: int,
    shift_i: int,
    max_int_i: int,
    min_int_i: int,
    double_round_i: bool,
    multiplier_i: int,
) -> int:
    """
    This function performs SIMD postprocessing of data given approximate algorithm of TOSA.rescale, with dynamically scaled shifts.
    """
    # Step 1: Subtract input zero point
    var_1 = data_in - input_zp_i

    # Additional Step 1:
    bits_to_shift_input = max(
        0, 9 + shift_i - int(np.ceil(np.log2(multiplier_i))) - 16
    )  # 8 can be adapted to be higher. higher will add more support for overflows, but will also reduce accuracy of the output.
    bits_to_shift_multiplier = max(0, int(np.ceil(np.log2(multiplier_i))) - 16)

    var_1 = var_1 >> bits_to_shift_input
    multiplier_i = multiplier_i >> bits_to_shift_multiplier
    shift_i = shift_i - bits_to_shift_input - bits_to_shift_multiplier

    # Step 2: Multiply with the multiplier avoiding overflow
    var_2 = np.int32(var_1) * np.int32(multiplier_i)

    # Step 3: Left shift one
    shifted_one = np.int32(1 << shift_i)

    # Step 4: Add shifted one
    var_3 = var_2 + shifted_one

    # Step 5: Double rounding
    if double_round_i:
        if var_1 > 0:
            var_4 = var_3 + np.int32(
                1 << (30 - bits_to_shift_multiplier - bits_to_shift_input)
            )
        else:
            var_4 = var_3 - np.int32(
                1 << (30 - bits_to_shift_multiplier - bits_to_shift_input)
            )
    else:
        # If double rounding is not used, we just pass the value through
        var_4 = var_3

    if shift_i > 31 - bits_to_shift_multiplier - bits_to_shift_input:
        var_5 = var_4
    else:
        var_5 = var_3

    # Step 6: Shift right
    var_6 = np.int32(var_5 >> shift_i)

    # Step 7: Add output zero point
    var_7 = var_6 + output_zp_i

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
    printer.print(rescale_down())
    with open(mlir_filename, "w") as output_file:
        output_file.write(output.getvalue())
