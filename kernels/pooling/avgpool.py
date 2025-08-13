import os
from io import StringIO

import numpy as np
from xdsl.builder import Builder
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    ModuleOp,
    TensorType,
    i8,
    i32,
    i64,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.tosa import AvgPool2DOp
from xdsl.printer import Printer


def avgpool(m: int = 28, n: int = 28, channels: int = 64):
    np.random.seed(42)  # For reproducibility
    # Define Variables For Program:
    a_type = TensorType(i8, (1, m, n, channels))
    # a_vals = np.full((channels, m, n), -8737248 , dtype=np.int32)
    a_vals = np.random.randint(
        -128,
        127,
        (m, n, channels),
        dtype=np.int8,
    )
    # print(a_vals)

    m_kernel = 3
    n_kernel = 3
    m_stride = 2
    n_stride = 2

    output_type = TensorType(
        i8,
        (1, (m - m_kernel) // m_stride + 1, (n - n_kernel) // n_stride + 1, channels),
    )
    golden_vals_list_intermediate = avgpool_golden(
        a_vals, m, n, channels, m_kernel, n_kernel, m_stride, n_stride
    )

    golden_vals_list = np.zeros(
        (
            ((m - m_kernel) // m_stride + 1),
            ((n - n_kernel) // n_stride + 1),
            channels,
        ),
        dtype=np.int8,
    )

    shift = 25  # TODO: figure out how these are actually determined
    multiplier = (2**shift) // (m_kernel * n_kernel)

    for i in range(0, ((m - m_kernel) // m_stride + 1)):
        for j in range(0, ((n - n_kernel) // n_stride + 1)):
            for k in range(channels):
                golden_vals_list[i, j, k] = golden_model_rescale_down(
                    golden_vals_list_intermediate[i, j, k],
                    0,
                    0,
                    shift,
                    127,
                    -128,
                    True,
                    multiplier,
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
        print(a_vals[:, :, 0])
        # print(a_vals[:,:, 1])
        # print(a_vals.flatten().tolist())
        golden = ConstantOp(
            DenseIntOrFPElementsAttr.from_list(
                output_type, golden_vals.flatten().tolist()
            )
        )
        print(golden_vals[:, :, 0])
        # print(golden_vals[:,:, 1])
        # print(golden_vals.flatten().tolist())

        # # Declare result tensor type
        # empty_tensor = EmptyOp([], output_type)

        # Specify the operation
        result = AvgPool2DOp(
            operands=[a.result],
            result_types=[output_type],
            properties={
                "kernel": DenseArrayBase.create_dense_int(i64, [3, 3]),
                "stride": DenseArrayBase.create_dense_int(i64, [2, 2]),
                "pad": DenseArrayBase.create_dense_int(i64, [0, 0, 0, 0]),
                "acc_type": i32,
            },
        )

        # Return both the computed result and the golden output
        ReturnOp(result, golden)

    function = FuncOp.from_region("snax_main", [], res_types, func_body)
    return ModuleOp([function])


def avgpool_golden(
    a_vals: np.ndarray,
    m: int,
    n: int,
    channels: int,
    m_kernel: int,
    n_kernel: int,
    m_stride: int,
    n_stride: int,
) -> np.ndarray:
    """
    Compute the golden output for maxpool operation.
    This function simulates the maxpool operation on the input tensor.
    """
    output = np.empty(
        (
            ((m - m_kernel) // m_stride + 1),
            ((n - n_kernel) // n_stride + 1),
            channels,
        ),
        dtype=np.int32,
    )
    # Iterate over each channel and apply max pooling
    for i in range(0, ((m - m_kernel) // m_stride + 1) * m_stride, m_stride):
        for j in range(0, ((n - n_kernel) // n_stride + 1) * n_stride, n_stride):
            for c in range(channels):
                # Extract the kernel region
                kernel_region = a_vals[i : i + m_kernel, j : j + n_kernel, c]
                # Compute the average value in the kernel region
                avg_value = np.sum(kernel_region)
                output[i // m_stride, j // n_stride, c] = avg_value
    return output


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
    This function performs SIMD postprocessing of data given approximate algorithm of TOSA.rescale,
    with dynamically scaled shifts.
    """
    # Step 1: Subtract input zero point
    var_1 = data_in - input_zp_i

    # Additional Step 1:
    bits_to_shift_input = max(0, 9 + shift_i - int(np.ceil(np.log2(multiplier_i))) - 16)
    # 8 can be adapted to be higher. higher will add more support for overflows,
    # but will also reduce accuracy of the output.
    bits_to_shift_multiplier = max(0, int(np.ceil(np.log2(multiplier_i))) - 16)

    var_1 = var_1 >> bits_to_shift_input
    multiplier_i = multiplier_i >> bits_to_shift_multiplier
    shift_i = shift_i - bits_to_shift_input - bits_to_shift_multiplier

    # Step 2: Multiply with the multiplier avoiding overflow
    var_2 = np.int32(var_1) * np.int32(multiplier_i)

    # Step 3: Left shift one
    shifted_one = np.int32(1 << (shift_i - 1))

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
    printer.print(avgpool())
    with open(mlir_filename, "w") as output_file:
        output_file.write(output.getvalue())
