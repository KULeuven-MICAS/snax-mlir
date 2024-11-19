from dataclasses import dataclass

import numpy as np
import torch.nn as nn
from torch import Tensor
from xdsl.builder import Builder
from xdsl.dialects import arith, builtin, func, linalg, tensor
from xdsl.dialects.builtin import i8, i32, i64
from xdsl.ir import BlockArgument, StringIO
from xdsl.parser import DenseIntOrFPElementsAttr
from xdsl.printer import Printer


@dataclass(frozen=True)
class ConvSpec:
    b: int
    ox: int
    oy: int
    fx: int
    fy: int
    c: int
    k: int
    groups: int = 1
    stride: int = 1
    dilation: int = 1


def generate_conv_ir(spec: ConvSpec, generate_constants: bool = True):
    # cop
    low_bound = 0
    high_bound = 10

    I_size = [spec.b, spec.c, (spec.oy * spec.stride) + spec.dilation*(spec.fy - 1), (spec.stride * spec.ox) + spec.dilation*(spec.fx - 1)]
    W_size = [spec.k, spec.c, spec.fy, spec.fx]

    if generate_constants:
        I = np.random.randint(low_bound, high_bound, size=I_size, dtype=np.dtype("int8"))
        input_vals: list[int] = I.flatten().tolist()
        W = np.random.randint(low_bound, high_bound, size=W_size, dtype=np.dtype("int8"))
        weight_vals: list[int] = W.flatten().tolist()

        conv = nn.Conv2d(spec.c, spec.k, spec.fx)
        conv.weight = nn.Parameter(Tensor(W))

        O = np.round(conv(Tensor(I)).detach().numpy()).astype(np.int32)
        output_vals: list[int] = O.flatten().tolist()

    else:
        input_vals = [0]
        output_vals = [0]
        weight_vals = [0]

    input_type = builtin.TensorType(i8, (spec.b, spec.c, (spec.oy * spec.stride) + (spec.fy - 1) * spec.dilation, (spec.ox * spec.stride) + (spec.fx - 1)*spec.dilation))
    kernel_type = builtin.TensorType(i8, (spec.k, spec.c, spec.fy, spec.fx))
    output_type = builtin.TensorType(i32, (spec.b, spec.k, spec.oy, spec.ox))

    # function returns golden output + computed output
    res_types = [output_type, output_type]

    dilations = DenseIntOrFPElementsAttr.tensor_from_list([spec.dilation], i64, [2])
    strides = DenseIntOrFPElementsAttr.tensor_from_list([spec.stride], i64, [2])

    @Builder.implicit_region([])
    def func_body(_) -> None:
        inputs = arith.Constant(DenseIntOrFPElementsAttr.from_list(input_type, input_vals), input_type)
        weights = arith.Constant(DenseIntOrFPElementsAttr.from_list(kernel_type, weight_vals), kernel_type)
        golden = arith.Constant(DenseIntOrFPElementsAttr.from_list(output_type, output_vals), output_type)
        empty_tensor = tensor.EmptyOp([], output_type)
        result = linalg.Conv2DNchwFchwOp(
            dilations, strides, (inputs.result, weights.result), (empty_tensor.results[0],)
        )
        func.Return(result, golden)

    function = func.FuncOp.from_region("conv", [], res_types, func_body)

    module = builtin.ModuleOp([function])

    return module


if __name__ == "__main__":
    spec = ConvSpec(1, 16, 16, 3, 3, 16, 16)

    output = StringIO()
    printer = Printer(stream=output)
    printer.print(generate_conv_ir(spec))
    with open('conv.mlir', "w") as output_file:
        output_file.write(output.getvalue())

