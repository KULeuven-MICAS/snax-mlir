import os
from io import StringIO

import numpy as np
from xdsl.builder import Builder
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    ModuleOp,
    TensorType,
    i8,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import MatmulOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.printer import Printer

"""
This file contains the implementation of a cascade matrix multiplication function

The arguments are:
    batch_size: int
    input_dim: int
    hidden_layers_dim: List[int]
    output_dim: int
"""


# TODO: This is just a simple scaling mechanism
# Just for the sake of making the matrices within int8
def scale_to_int8(arr):
    scaled = np.right_shift(arr, 9)
    scaled = np.clip(scaled, -128, 127)
    return scaled.astype(np.int8)


def matmul(m=16, n=16, k=16):
    # Define Variables For Program:

    np.random.seed(2)  # For reproducibility

    a_type = TensorType(i8, (m, k))
    a_vals = np.random.randint(-128, 127, (m, k))

    b_type = TensorType(i8, (k, n))
    b_vals = np.random.randint(-128, 127, (k, n))

    output_type = TensorType(i8, (m, n))
    golden_vals = a_vals @ b_vals
    golden_vals = scale_to_int8(golden_vals)

    res_types = [output_type] * 2

    # Define Program:

    @Builder.implicit_region([])
    def func_body(_) -> None:
        # Declare constants
        a = ConstantOp(DenseIntOrFPElementsAttr.from_list(a_type, a_vals.flatten().tolist()))
        b = ConstantOp(DenseIntOrFPElementsAttr.from_list(b_type, b_vals.flatten().tolist()))
        golden = ConstantOp(DenseIntOrFPElementsAttr.from_list(output_type, golden_vals.flatten().tolist()))

        # Declare result tensor type
        empty_tensor = EmptyOp([], output_type)

        # Specify the operation
        result = MatmulOp([a.result, b.result], empty_tensor.results)

        # Return both the computed result and the golden output
        ReturnOp(result, golden)

    function = FuncOp.from_region("snax_main", [], res_types, func_body)
    return ModuleOp([function])


if __name__ == "__main__":
    # Get the name of the current Python script and replace its extension with .mlir
    script_name = os.path.basename(__file__)
    mlir_filename = os.path.splitext(script_name)[0] + ".mlir"

    # Generate IR and write it to the specified MLIR file
    output = StringIO()
    printer = Printer(stream=output)
    printer.print(matmul())
    with open(mlir_filename, "w") as output_file:
        output_file.write(output.getvalue())
