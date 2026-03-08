"""Generate MLIR for a quantized int8 matrix multiplication."""

import argparse
from io import StringIO

import numpy as np
from xdsl.builder import Builder
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    ModuleOp,
    TensorType,
    i8,
    i32,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import QuantizedMatmulOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.printer import Printer


def gemm(m=16, n=16, k=16):
    a_type = TensorType(i8, (m, k))
    a_vals = np.random.randint(-128, 127, (m, k))

    b_type = TensorType(i8, (k, n))
    b_vals = np.random.randint(-128, 127, (k, n))

    output_type = TensorType(i32, (m, n))
    c_vals = np.zeros((m, n), dtype=np.int32)

    golden_vals = a_vals @ b_vals + c_vals

    res_types = [output_type] * 2

    @Builder.implicit_region([])
    def func_body(_) -> None:
        a = ConstantOp(
            DenseIntOrFPElementsAttr.from_list(a_type, a_vals.flatten().tolist())
        )
        b = ConstantOp(
            DenseIntOrFPElementsAttr.from_list(b_type, b_vals.flatten().tolist())
        )
        c = ConstantOp(
            DenseIntOrFPElementsAttr.from_list(output_type, c_vals.flatten().tolist())
        )
        golden = ConstantOp(
            DenseIntOrFPElementsAttr.from_list(
                output_type, golden_vals.flatten().tolist()
            )
        )

        c0 = ConstantOp.from_int_and_width(0, 32)

        result = QuantizedMatmulOp(
            [a.result, b.result, c0.result, c0.result], [c.result]
        )

        ReturnOp(result, golden)

    function = FuncOp.from_region("snax_main", [], res_types, func_body)
    return ModuleOp([function])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MLIR for a quantized matmul"
    )
    parser.add_argument(
        "--m", type=int, default=16, help="Number of rows of matrix A and result"
    )
    parser.add_argument(
        "--n", type=int, default=16, help="Number of columns of matrix B and result"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=16,
        help="Number of columns of matrix A / rows of matrix B",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output MLIR file path",
    )

    args = parser.parse_args()

    output = StringIO()
    printer = Printer(stream=output)
    printer.print(gemm(m=args.m, n=args.n, k=args.k))

    with open(args.output, "w") as output_file:
        output_file.write(output.getvalue())
