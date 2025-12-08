import argparse
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

from snaxc.dialects.kernel import RescaleOp
from util.gemmx.simd_golden_model import postprocessing_simd_golden_model


def gemm(m=64, n=64, k=64):
    # Define Variables For Program:

    a_type = TensorType(i8, (m, k))
    a_vals = np.random.randint(-128, 127, (m, k))

    b_type = TensorType(i8, (k, n))
    b_vals = np.random.randint(-128, 127, (k, n))

    c_type = TensorType(i32, (m, n))
    golden_vals = a_vals @ b_vals

    res_types = [c_type] * 2

    # Define Program:

    @Builder.implicit_region([])
    def func_body(_) -> None:
        # Declare constants
        a = ConstantOp(DenseIntOrFPElementsAttr.from_list(a_type, a_vals.flatten().tolist()))
        b = ConstantOp(DenseIntOrFPElementsAttr.from_list(b_type, b_vals.flatten().tolist()))
        golden = ConstantOp(DenseIntOrFPElementsAttr.from_list(c_type, golden_vals.flatten().tolist()))

        c0 = ConstantOp.from_int_and_width(0, 32)

        # Declare result tensor type
        empty_tensor = EmptyOp([], c_type)

        # Specify the operation
        result = QuantizedMatmulOp([a.result, b.result, c0.result, c0.result], empty_tensor.results)

        # Return both the computed result and the golden output
        ReturnOp(result, golden)

    function = FuncOp.from_region("snax_main", [], res_types, func_body)
    return ModuleOp([function])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MLIR for a quantized matmul")
    parser.add_argument(
        "--m", type=int, default=64, help="Number of rows of matrix A and result"
    )
    parser.add_argument(
        "--n", type=int, default=64, help="Number of columns of matrix B and result"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=64,
        help="Number of columns of matrix A / rows of matrix B",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output MLIR file path"
    )

    args = parser.parse_args()

    output = StringIO()
    printer = Printer(stream=output)
    printer.print(gemm(m=args.m, n=args.n, k=args.k))

    with open(args.output, "w") as output_file:
        output_file.write(output.getvalue())
