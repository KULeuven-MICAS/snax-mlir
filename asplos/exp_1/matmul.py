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

    output_type = TensorType(i8, (m, n))
    golden_vals = a_vals @ b_vals
    golden_vals = postprocessing_simd_golden_model(
        golden_vals,
        input_zp_i=13,
        output_zp_i=-27,
        shift_i=39,
        max_int_i=127,
        min_int_i=-128,
        double_round_i=True,
        multiplier_i=1234567890,
    )

    res_types = [output_type] * 2

    # Define Program:

    @Builder.implicit_region([])
    def func_body(_) -> None:
        # Declare constants
        a = ConstantOp(DenseIntOrFPElementsAttr.from_list(a_type, a_vals.flatten().tolist()))
        b = ConstantOp(DenseIntOrFPElementsAttr.from_list(b_type, b_vals.flatten().tolist()))
        golden = ConstantOp(DenseIntOrFPElementsAttr.from_list(output_type, golden_vals.flatten().tolist()))

        c0 = ConstantOp.from_int_and_width(0, 32)

        # Declare result tensor type
        empty_tensor = EmptyOp([], c_type)

        # Specify the operation
        result = QuantizedMatmulOp([a.result, b.result, c0.result, c0.result], empty_tensor.results)

        # Rescale:
        empty_tensor_3 = EmptyOp([], output_type)

        arg_types = [i32, i8]

        @Builder.implicit_region(arg_types)
        def init_body(args: tuple[BlockArgument, ...]) -> None:
            rescaled = RescaleOp(args[0], i8, 13, -27, [1234567890], [39], 127, -128, True)
            YieldOp(rescaled)

        indexing_maps = [AffineMapAttr(AffineMap.from_callable(lambda x, y: (x, y)))] * 2

        rescaled = GenericOp(
            [result.results[0]],
            [empty_tensor_3.tensor],
            init_body,
            indexing_maps,
            [IteratorTypeAttr.parallel()] * 2,
            [output_type],
        )

        # Return both the computed result and the golden output
        ReturnOp(rescaled, golden)

    function = FuncOp.from_region("snax_main", [], res_types, func_body)
    return ModuleOp([function])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MLIR for a quantized matmul")
    parser.add_argument("--m", type=int, default=64, help="Number of rows of matrix A and result")
    parser.add_argument("--n", type=int, default=64, help="Number of columns of matrix B and result")
    parser.add_argument("--k", type=int, default=64, help="Number of columns of matrix A / rows of matrix B")
    parser.add_argument("--output", type=str, required=True, help="Output MLIR file path")

    args = parser.parse_args()

    output = StringIO()
    printer = Printer(stream=output)
    printer.print(gemm(m=args.m, n=args.n, k=args.k))

    with open(args.output, "w") as output_file:
        output_file.write(output.getvalue())
