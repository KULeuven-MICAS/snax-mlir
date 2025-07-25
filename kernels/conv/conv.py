import os
from dataclasses import dataclass
from io import StringIO
from typing import Any, cast

import numpy as np
import tensorflow as tf
from numpy._typing import NDArray
from xdsl.builder import Builder
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    ModuleOp,
    TensorType,
    i8,
    i32,
    i64,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import Conv2DNchwFchwOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.printer import Printer


@dataclass(frozen=True)
class ConvSpec:
    b: int  # Batch size
    ox: int  # Output width
    oy: int  # Output height
    fx: int  # Filter width
    fy: int  # Filter height
    c: int  # Input channels
    k: int  # Output channels
    groups: int = 1  # Number of groups
    stride: int = 1  # Stride
    dilation: int = 1  # Dilation


def generate_conv_tensors(spec: ConvSpec) -> tuple[NDArray[np.int8], NDArray[np.int8]]:
    """
    Generate input and weight tensors based on the given convolution specification.
    """
    # Compute the required input size (ensuring no padding is needed)
    ix = (spec.ox - 1) * spec.stride + (spec.fx - 1) * spec.dilation + 1
    iy = (spec.oy - 1) * spec.stride + (spec.fy - 1) * spec.dilation + 1

    i_size = (spec.b, iy, ix, spec.c)
    w_size = (spec.fy, spec.fx, spec.c, spec.k)

    # Create input tensor (batch_size, height, width, channels) = NHWC
    input_tensor = np.random.randint(-128, 127, i_size, dtype=np.int8)

    # Create weight tensor (filter_height, filter_width, in_channels_per_group, out_channels) = HWCF
    weight_tensor = np.random.randint(-128, 127, w_size, dtype=np.int8)

    return input_tensor, weight_tensor


def compute_convolution(spec: ConvSpec, input_tensor: NDArray[np.int8], weight_tensor: NDArray[np.int8]):
    """
    Perform convolution using TensorFlow.
    """
    # Convert numpy arrays to TensorFlow tensors
    input_tf = tf.convert_to_tensor(input_tensor, dtype=tf.int8)
    weight_tf = tf.convert_to_tensor(weight_tensor, dtype=tf.int8)

    # Perform convolution (int32 output to avoid overflow)
    output_tf = tf.nn.conv2d(
        input=tf.cast(input_tf, tf.int32),
        filters=tf.cast(weight_tf, tf.int32),
        strides=[1, spec.stride, spec.stride, 1],
        padding="VALID",  # No padding
        dilations=[1, spec.dilation, spec.dilation, 1],
    )

    return output_tf.numpy()


def conv(spec: ConvSpec):
    input, weight = generate_conv_tensors(spec)
    output = compute_convolution(spec, input, weight)

    # reshape to the mlir conv2d op spec
    input = input.transpose((0, 3, 1, 2))  # NHWC -> NCHW
    weight = weight.transpose((3, 2, 0, 1))  # HWCF -> FCHW
    output = output.transpose((0, 3, 1, 2))  # NHWC -> NCHW

    input_type = TensorType(i8, shape=input.shape)
    weight_type = TensorType(i8, shape=weight.shape)
    output_type = TensorType(i32, shape=output.shape)

    res_types = [output_type, output_type]

    dilations = DenseIntOrFPElementsAttr.tensor_from_list([spec.dilation], i64, [2])
    strides = DenseIntOrFPElementsAttr.tensor_from_list([spec.stride], i64, [2])

    # Define Program:

    @Builder.implicit_region([])
    def func_body(_) -> None:
        # Declare constants
        input_c = ConstantOp(DenseIntOrFPElementsAttr.from_list(input_type, input.flatten().tolist()))
        weight_c = ConstantOp(DenseIntOrFPElementsAttr.from_list(weight_type, weight.flatten().tolist()))
        golden_c = ConstantOp(DenseIntOrFPElementsAttr.from_list(output_type, output.flatten().tolist()))

        # Declare result tensor type
        empty_tensor = EmptyOp([], output_type)

        # Specify the operation
        result = Conv2DNchwFchwOp(
            (input_c.result, weight_c.result),
            (empty_tensor.results[0],),
            (output_type,),
            {"dilations": dilations, "strides": strides},
        )

        ReturnOp(result, golden_c)

    function = FuncOp.from_region("snax_main", [], res_types, func_body)
    return ModuleOp([function])


if __name__ == "__main__":
    # Check if running under Snakemake
    snakemake_instance = globals().get("snakemake", None)

    if snakemake_instance:
        cfg = cast(dict[str, Any], snakemake.config["convspecs"][snakemake.wildcards[0]])  # noqa: F821
        spec = ConvSpec(
            b=cfg["b"],
            ox=cfg["ox"],
            oy=cfg["oy"],
            fx=cfg["fx"],
            fy=cfg["fy"],
            c=cfg["c"],
            k=cfg["k"],
            groups=cfg.get("groups", 1),
            stride=cfg.get("stride", 1),
            dilation=cfg.get("dilation", 1),
        )
        output_file_path = snakemake.output[0]  # noqa: F821
    else:
        # Standalone default
        spec = ConvSpec(1, 16, 16, 3, 3, 16, 16)
        spec = ConvSpec(1, 8, 8, 3, 3, 3, 8)
        script_name = os.path.basename(__file__)
        output_file_path = os.path.splitext(script_name)[0] + ".mlir"

    # Generate and write MLIR
    output = StringIO()
    printer = Printer(stream=output)
    printer.print(conv(spec))
    with open(output_file_path, "w") as output_file:
        output_file.write(output.getvalue())
