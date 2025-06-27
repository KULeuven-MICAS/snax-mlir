import os
from io import StringIO

import numpy as np
from xdsl.builder import Builder
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    AffineMapAttr,
    DenseIntOrFPElementsAttr,
    ModuleOp,
    TensorType,
    i8,
    i32,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import GenericOp, IteratorTypeAttr, QuantizedMatmulOp, YieldOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import BlockArgument
from xdsl.ir.affine import AffineMap
from xdsl.printer import Printer


def gemm(m=16, n=16, k=16):
    # Define Variables For Program:

    a_type = TensorType(i8, (m, k))
    a_vals = np.random.randint(-128, 127, (m, k))

    b_type = TensorType(i8, (k, n))
    b_vals = np.random.randint(-128, 127, (k, n))

    c_type = TensorType(i32, (n,))
    c_vals = np.random.randint(-1024, 1023, (n,))

    output_type = TensorType(i32, (m, n))

    golden_vals = a_vals @ b_vals + c_vals

    res_types = [output_type] * 2

    # Define Program:

    @Builder.implicit_region([])
    def func_body(_) -> None:
        # Declare constants
        a = ConstantOp(DenseIntOrFPElementsAttr.from_list(a_type, a_vals.flatten().tolist()))
        b = ConstantOp(DenseIntOrFPElementsAttr.from_list(b_type, b_vals.flatten().tolist()))
        c = ConstantOp(DenseIntOrFPElementsAttr.from_list(c_type, c_vals.flatten().tolist()))
        golden = ConstantOp(DenseIntOrFPElementsAttr.from_list(output_type, golden_vals.flatten().tolist()))

        c0 = ConstantOp.from_int_and_width(0, 32)

        # Declare result tensor type
        empty_tensor = EmptyOp([], output_type)

        # create init using a linalg yield
        arg_types = [output_type.element_type] * 2

        @Builder.implicit_region(arg_types)
        def init_body(args: tuple[BlockArgument, ...]) -> None:
            YieldOp(args[0])

        indexing_maps = [
            AffineMapAttr(AffineMap.from_callable(lambda _, y: (y,))),
            AffineMapAttr(AffineMap.from_callable(lambda x, y: (x, y))),
        ]

        init = GenericOp(
            [c.result],
            [empty_tensor.tensor],
            init_body,
            indexing_maps,
            [IteratorTypeAttr.parallel()] * 2,
            [output_type],
        )

        # Specify the operation, starting from an init
        result = QuantizedMatmulOp([a.result, b.result, c0.result, c0.result], [init.res[0]])

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
    printer.print(gemm())
    with open(mlir_filename, "w") as output_file:
        output_file.write(output.getvalue())
