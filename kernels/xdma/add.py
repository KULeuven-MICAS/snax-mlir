import os
from io import StringIO

import numpy as np
from xdsl.builder import Builder
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    ModuleOp,
    TensorType,
    i32,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import AddOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.printer import Printer


def add(n=128):
    # Define Variables For Program:
    a_type = TensorType(i32, (n,))
    a_vals = np.random.randint(-128, 127, (n,))

    b_type = TensorType(i32, (n,))
    b_vals = np.random.randint(-128, 127, (n,))

    output_type = TensorType(i32, (n,))
    golden_vals = a_vals + b_vals

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
        result = AddOp([a.result, b.result], empty_tensor.results, [output_type])

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
    printer.print(add())
    with open(mlir_filename, "w") as output_file:
        output_file.write(output.getvalue())
