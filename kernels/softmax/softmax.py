import math
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
from xdsl.dialects.linalg import GenericOp, IteratorTypeAttr, YieldOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import BlockArgument
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap
from xdsl.printer import Printer

from snaxc.dialects.kernel import SoftMaxOp


def softmax(m: int = 43, channels: int = 64):
    np.random.seed(42)  # For reproducibility
    # Define Variables For Program:
    scaling_factor = 10000
    a_type = TensorType(i32, (m, channels))
    # a_vals = np.full((channels, m, n), -8737248 , dtype=np.int32)
    a_vals = np.random.randint(
        -40000,
        40000,
        (m, channels),
        dtype=np.int32,
    )

    kernel_type = TensorType(i8, (m, 1))

    output_type = TensorType(i32, (m, channels))
    golden_vals = np.empty_like(a_vals, dtype=np.int32)
    for i in range(channels):
        golden_vals[:, i] = integer_softmax(
            a_vals[:, i],
            scaling_factor_exp=scaling_factor,
        )
    res_types = [output_type] * 2

    # Define Program:

    @Builder.implicit_region([])
    def func_body(_) -> None:
        # Declare constants
        input_matrix = ConstantOp(
            DenseIntOrFPElementsAttr.from_list(a_type, a_vals.flatten().tolist())
        )

        golden = ConstantOp(
            DenseIntOrFPElementsAttr.from_list(
                output_type, golden_vals.flatten().tolist()
            )
        )

        empty_kernel = EmptyOp([], kernel_type)
        empty_output = EmptyOp([], output_type)

        arg_types = [i32, i8, i32]

        indexing_maps = [
            # Input tensor map: (d0, ((d1 * 2) + d4), ((d2 * 2) + d5), d3)
            AffineMapAttr(
                AffineMap(
                    3,
                    0,
                    (
                        AffineDimExpr(1),  # d1
                        AffineDimExpr(0) + 16 * AffineDimExpr(2),  # (d0 + (16 * d2))
                    ),
                )
            ),
            # Kernel tensor map: (d4, d5)
            AffineMapAttr(
                AffineMap(
                    3,
                    0,
                    (
                        AffineConstantExpr(1),  # d4
                        AffineDimExpr(0) + 16 * AffineDimExpr(2),  # d5
                    ),
                )
            ),
            # Output tensor map: (d0, d1, d2, d3)
            AffineMapAttr(
                AffineMap(
                    3,
                    0,
                    (
                        AffineDimExpr(1),  # d1
                        AffineDimExpr(0) + 16 * AffineDimExpr(2),  # (d0 + (16 * d2))
                    ),
                )
            ),
        ]

        @Builder.implicit_region(arg_types)
        def init_body(args: tuple[BlockArgument, ...]) -> None:
            # Specify the operation
            result = SoftMaxOp(
                args[0],
                output_type=output_type,
                scaling_factor=scaling_factor,
            )
            YieldOp(result)

        result = GenericOp(
            [input_matrix.result, empty_kernel.tensor],
            [empty_output.tensor],
            init_body,
            indexing_maps,
            [IteratorTypeAttr.parallel()] * 3,
            [output_type],
        )

        # Return both the computed result and the golden output
        ReturnOp(result, golden)

    function = FuncOp.from_region("snax_main", [], res_types, func_body)
    return ModuleOp([function])


def find_max(array: np.ndarray):
    """Find the maximum value in an array."""
    max_value = array[0]
    for value in array:
        if value > max_value:
            max_value = value
    return max_value


def subtract_max(array: np.ndarray, max_value: np.int32):
    """Subtract the maximum value from each element in the array."""
    new_array = np.empty_like(array, dtype=np.int32)
    for i in range(len(array)):
        if (int(array[i]) - int(max_value)) < np.iinfo(np.int32).min:
            new_array[i] = np.iinfo(np.int32).min
        else:
            new_array[i] = array[i] - max_value
    return new_array


def integer_poly(
    x: np.int32, inverse_scaling_factor: int, a: float, b: float, c: float
):
    a_scaled = int(a * inverse_scaling_factor)
    b_scaled = int(b * inverse_scaling_factor)
    c_scaled = int(c * (inverse_scaling_factor**3)) >> math.floor(
        math.log2(inverse_scaling_factor) * 2
    )

    output = np.int32(
        (
            (a_scaled * (int(x) + b_scaled) ** 2)
            >> math.floor(math.log2(inverse_scaling_factor) * 2)
        )
        + c_scaled
    )

    scaling_factor_out = (inverse_scaling_factor**3) >> math.floor(
        math.log2(inverse_scaling_factor) * 2
    )

    return output, scaling_factor_out


def integer_exp(array: np.ndarray, inverse_scaling_factor: int):
    """Calculate the exponential of each element in the array."""
    exp_array = np.empty_like(array, dtype=np.int32)
    a = 0.3585
    b = 1.353
    c = 0.344
    q_ln2 = int(math.log(2) * inverse_scaling_factor)
    for i in range(len(array)):
        z = math.floor(-array[i] / q_ln2)
        q_p = array[i] + z * q_ln2
        q_l, scaling_factor_exp = integer_poly(
            q_p,
            inverse_scaling_factor,
            a,
            b,
            c,
        )
        if z < 16:
            exp_array[i] = int(int(q_l) >> z)
        else:
            exp_array[i] = 0
    return exp_array, scaling_factor_exp


def integer_softmax(array: np.ndarray, scaling_factor_exp: int):
    max = find_max(array)
    array = subtract_max(array, max)
    array, scaling_factor_exp = integer_exp(array, scaling_factor_exp)
    sum_exp = np.sum(array)
    divider = (2**32 - 1) // sum_exp
    array_out = array * divider
    return array_out


if __name__ == "__main__":
    # Get the name of the current Python script and replace its extension with .mlir
    script_name = os.path.basename(__file__)
    mlir_filename = os.path.splitext(script_name)[0] + ".mlir"

    # Generate IR and write it to the specified MLIR file
    output = StringIO()
    printer = Printer(stream=output)
    printer.print(softmax())
    with open(mlir_filename, "w") as output_file:
        output_file.write(output.getvalue())
