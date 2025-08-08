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
    i64,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.tosa import MaxPool2DOp
from xdsl.printer import Printer


def maxpool(m: int = 28, n: int = 28, channels: int = 64):
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
    golden_vals_list = maxpool_golden(a_vals, m, n, channels, m_kernel, n_kernel, m_stride, n_stride)

    golden_vals = np.array(golden_vals_list, dtype=np.int8)

    res_types = [output_type] * 2

    # Define Program:

    @Builder.implicit_region([])
    def func_body(_) -> None:
        # Declare constants
        a = ConstantOp(DenseIntOrFPElementsAttr.from_list(a_type, a_vals.flatten().tolist()))
        print(a_vals[:, :, 0])
        # print(a_vals[:,:, 1])
        # print(a_vals.flatten().tolist())
        golden = ConstantOp(DenseIntOrFPElementsAttr.from_list(output_type, golden_vals.flatten().tolist()))
        print(golden_vals[:, :, 0])
        # print(golden_vals[:,:, 1])
        # print(golden_vals.flatten().tolist())

        # # Declare result tensor type
        # empty_tensor = EmptyOp([], output_type)

        # Specify the operation
        result = MaxPool2DOp(
            operands=[a.result],
            result_types=[output_type],
            properties={
                "kernel": DenseArrayBase.create_dense_int(i64, [3, 3]),
                "stride": DenseArrayBase.create_dense_int(i64, [2, 2]),
                "padding": DenseArrayBase.create_dense_int(i64, [0, 0, 0, 0]),
            },
        )

        # Return both the computed result and the golden output
        ReturnOp(result, golden)

    function = FuncOp.from_region("snax_main", [], res_types, func_body)
    return ModuleOp([function])


def maxpool_golden(
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
        dtype=np.int8,
    )
    # Iterate over each channel and apply max pooling
    for i in range(0, ((m - m_kernel) // m_stride + 1) * m_stride, m_stride):
        for j in range(0, ((n - n_kernel) // n_stride + 1) * n_stride, n_stride):
            for c in range(channels):
                # Extract the kernel region
                kernel_region = a_vals[i : i + m_kernel, j : j + n_kernel, c]
                # Compute the maximum value in the kernel region
                max_value = np.max(kernel_region)
                output[i // m_stride, j // n_stride, c] = max_value
    return output


if __name__ == "__main__":
    # Get the name of the current Python script and replace its extension with .mlir
    script_name = os.path.basename(__file__)
    mlir_filename = os.path.splitext(script_name)[0] + ".mlir"

    # Generate IR and write it to the specified MLIR file
    output = StringIO()
    printer = Printer(stream=output)
    printer.print(maxpool())
    with open(mlir_filename, "w") as output_file:
        output_file.write(output.getvalue())
