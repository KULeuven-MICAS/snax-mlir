from collections.abc import Sequence
from dataclasses import dataclass
from io import StringIO

import numpy as np
import tensorflow as tf
import yaml
from dacite import from_dict
from numpy._typing import NDArray
from xdsl.builder import Builder
from xdsl.dialects import builtin, transform
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    IntegerType,
    ModuleOp,
    TensorType,
    UnitAttr,
    i8,
    i64,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import Conv2DNchwFchwOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import SSAValue
from xdsl.printer import Printer

from asplos.util.convspecs import TiledConfig


def scale_to_int8(arr):
    scaled = np.right_shift(arr, 9)
    return scaled.astype(np.int8)


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

    np.random.seed(0)

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


def tiled_conv(spec: ConvSpec, tile_k: int, tile_y: int):
    input, weight = generate_conv_tensors(spec)
    output = compute_convolution(spec, input, weight)

    # reshape to the mlir conv2d op spec
    input = input.transpose((0, 3, 1, 2))  # NHWC -> NCHW
    weight = weight.transpose((3, 2, 0, 1))  # HWCF -> FCHW
    output = output.transpose((0, 3, 1, 2))  # NHWC -> NCHW

    golden_vals = scale_to_int8(output)

    input_type = TensorType(i8, shape=input.shape)
    weight_type = TensorType(i8, shape=weight.shape)
    output_type = TensorType(i8, shape=output.shape)

    res_types = [output_type, output_type]

    dilations = DenseIntOrFPElementsAttr.tensor_from_list([spec.dilation], i64, [2])
    strides = DenseIntOrFPElementsAttr.tensor_from_list([spec.stride], i64, [2])

    # Define Program:

    @Builder.implicit_region([])
    def func_body(_) -> None:
        # Declare constants
        input_c = ConstantOp(DenseIntOrFPElementsAttr.from_list(input_type, input.flatten().tolist()))
        weight_c = ConstantOp(DenseIntOrFPElementsAttr.from_list(weight_type, weight.flatten().tolist()))
        golden_c = ConstantOp(DenseIntOrFPElementsAttr.from_list(output_type, golden_vals.flatten().tolist()))

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

    # Manually speficy tiling sequence
    @Builder.implicit_region([transform.AnyOpType()])
    def tiling_sequence(args: Sequence[SSAValue]):
        matmul_op = transform.MatchOp(args[0], ["linalg.conv_2d_nchw_fchw"])
        transform.TileOp(
            target=matmul_op.result,
            dynamic_sizes=[],
            scalable_sizes=DenseArrayBase.create_dense_int(IntegerType(1), [0, 0, 0, 0]),
            static_sizes=DenseArrayBase.create_dense_int(
                IntegerType(64), [1, spec.k // tile_k, spec.oy // tile_y, spec.ox]
            ),
        )
        transform.YieldOp()

    function_type = builtin.FunctionType.from_lists([transform.AnyOpType()], [])
    transform_sequence = transform.NamedSequenceOp("__transform_main", function_type, tiling_sequence)

    return ModuleOp([function, transform_sequence], {"transform.with_named_sequence": UnitAttr()})


if __name__ == "__main__":
    # load in tiled layers config:
    with open("tiled_resnet_layers.yaml") as f:
        yaml_data = yaml.safe_load(f)

    tiled_config = from_dict(data_class=TiledConfig, data=yaml_data)

    for layer in tiled_config.layers:
        spec = ConvSpec(
            b=1,
            ox=layer.tiled_ox,
            oy=layer.tiled_oy * layer.tile_y,
            fx=layer.layer.fx,
            fy=layer.layer.fy,
            c=layer.tiled_c,
            k=layer.tiled_k * layer.tile_k,
            groups=1,
            stride=layer.layer.stride,
            dilation=1,
        )

        # Generate and write MLIR
        output = StringIO()
        printer = Printer(stream=output)
        printer.print(tiled_conv(spec, layer.tile_k, layer.tile_y))
        with open(layer.layer.name + ".mlir", "w") as output_file:
            output_file.write(output.getvalue())
