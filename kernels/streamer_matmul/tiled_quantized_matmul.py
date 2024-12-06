import os
from io import StringIO

import numpy as np
from xdsl.builder import Builder
from xdsl.dialects import builtin, transform
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    ModuleOp,
    TensorType,
    UnitAttr,
    i8,
    i32,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import QuantizedMatmulOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.parser import DenseArrayBase, IntegerType
from xdsl.printer import Printer


def matmul(m=16, n=16, k=16):
    # Define Variables For Program:

    a_type = TensorType(i8, (m, k))
    a_vals = np.random.randint(-127, 128, (m, k))

    b_type = TensorType(i8, (k, n))
    b_vals = np.random.randint(-127, 128, (k, n))

    output_type = TensorType(i32, (m, n))
    golden_vals = a_vals @ b_vals

    res_types = [output_type] * 2

    # Define Program:
    @Builder.implicit_region([])
    def func_body(_) -> None:
        # Declare constants
        a = ConstantOp(
            DenseIntOrFPElementsAttr.from_list(a_type, a_vals.flatten().tolist())
        )
        b = ConstantOp(
            DenseIntOrFPElementsAttr.from_list(b_type, b_vals.flatten().tolist())
        )
        golden = ConstantOp(
            DenseIntOrFPElementsAttr.from_list(
                output_type, golden_vals.flatten().tolist()
            )
        )

        c0 = ConstantOp.from_int_and_width(0, 32)

        # Declare result tensor type
        empty_tensor = EmptyOp([], output_type)

        # Specify the operation
        result = QuantizedMatmulOp(
            (a.result, b.result, c0.result, c0.result), empty_tensor.results
        )

        # Return both the computed result and the golden output
        ReturnOp(result, golden)

    function = FuncOp.from_region("snax_main", [], res_types, func_body)

    # Manually speficy tiling sequence
    transform_inputs = [
        transform.AnyOpType(),
        transform.OperationType("linalg.quantized_matmul"),
    ]

    @Builder.implicit_region(transform_inputs)
    def tiling_sequence(args):
        transform.TileOp(
            target=args[1],
            dynamic_sizes=[],
            scalable_sizes=DenseArrayBase.create_dense_int(IntegerType(1), [0, 0]),
            static_sizes=DenseArrayBase.create_dense_int(IntegerType(64), [8, 8]),
        )

        transform.YieldOp()

    function_type = builtin.FunctionType.from_lists(transform_inputs, [])
    transform_sequence = transform.NamedSequenceOp(
        "__transform_main", function_type, tiling_sequence
    )

    return ModuleOp(
        [function, transform_sequence], {"transform.with_named_sequence": UnitAttr()}
    )


if __name__ == "__main__":
    # Get the name of the current Python script and replace its extension with .mlir
    script_name = os.path.basename(__file__)
    mlir_filename = os.path.splitext(script_name)[0] + ".transform.mlir"

    # Generate IR and write it to the specified MLIR file
    output = StringIO()
    printer = Printer(stream=output)
    printer.print(matmul())
    with open(mlir_filename, "w") as output_file:
        output_file.write(output.getvalue())
