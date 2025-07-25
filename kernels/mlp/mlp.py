import os
from io import StringIO

import numpy as np
from xdsl.builder import Builder
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    ModuleOp,
    TensorType,
    i8,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import MatmulOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.printer import Printer

"""
This file contains the implementation of a cascade matrix multiplication function

The arguments are:
    batch_size: int
    input_dim: int
    hidden_layers_dim: List[int]
    output_dim: int
"""


# TODO: This is just a simple scaling mechanism
# Just for the sake of making the matrices within int8
def scale_to_int8(arr):
    scaled = np.right_shift(arr, 9)
    return scaled.astype(np.int8)


# Cascade matrix multiplication function
# Technically a multilayer perceptron (MLP) with multiple hidden layers
def cascade_matmul(batch_size, input_dim, hidden_layers_dim, output_dim):
    # Input tensor
    np.random.seed(0)  # For reproducibility
    input_vals = np.random.randint(-128, 127, (batch_size, input_dim))

    # Iterate through different hidden layers
    hidden_weights_list = []
    hidden_output_vals = []

    for i in range(len(hidden_layers_dim)):
        # Generate random weights for the current layer
        # then perform matrix multiplication
        # and scale the result to int8
        if i == 0:
            hidden_weights_list.append(np.random.randint(-128, 127, (input_dim, hidden_layers_dim[i])))
            # Perform matrix multiplication for the first layer
            vXM = input_vals @ hidden_weights_list[i]
            hidden_output_vals.append(scale_to_int8(vXM))
        else:
            hidden_weights_list.append(np.random.randint(-128, 127, (hidden_layers_dim[i - 1], hidden_layers_dim[i])))
            vXM = hidden_output_vals[i - 1] @ hidden_weights_list[i]
            hidden_output_vals.append(scale_to_int8(vXM))

    # Generate random weights for the output layer
    output_weights = np.random.randint(-128, 127, (hidden_layers_dim[-1], output_dim))
    vXM = hidden_output_vals[-1] @ output_weights
    # Perform matrix multiplication for the output layer
    output_vals = scale_to_int8(vXM)

    return (
        input_vals,
        hidden_weights_list,
        hidden_output_vals,
        output_weights,
        output_vals,
    )


# Defining types and paramters for the MLIR generation
def mlir_cascade_matmul(input_vals, hidden_weights_list, hidden_output_vals, output_weights, output_vals):
    # Define types For Program:
    # For the input weights
    input_type = TensorType(i8, input_vals.shape)

    # For the hidden weights
    hidden_weights_types = []
    hidden_output_types = []
    for i in range(len(hidden_weights_list)):
        hidden_weights_types.append(TensorType(i8, hidden_weights_list[i].shape))
        hidden_output_types.append(TensorType(i8, hidden_output_vals[i].shape))

    # For the output weights
    output_weights_type = TensorType(i8, output_weights.shape)

    # For the output values
    output_type = TensorType(i8, output_vals.shape)

    # Result types
    res_types = [output_type] * 2

    # Define Program:

    @Builder.implicit_region([])
    def func_body(_) -> None:
        # Declare constants
        # For the input weights
        input_const = ConstantOp(DenseIntOrFPElementsAttr.from_list(input_type, input_vals.flatten().tolist()))

        # For the hidden weights
        hidden_weight_const = []
        for i in range(len(hidden_weights_list)):
            hidden_weight_const.append(
                ConstantOp(
                    DenseIntOrFPElementsAttr.from_list(
                        hidden_weights_types[i],
                        hidden_weights_list[i].flatten().tolist(),
                    )
                )
            )

        # For the output weights
        output_weight_const = ConstantOp(
            DenseIntOrFPElementsAttr.from_list(output_weights_type, output_weights.flatten().tolist())
        )

        # For the output values
        output_const = ConstantOp(DenseIntOrFPElementsAttr.from_list(output_type, output_vals.flatten().tolist()))

        # TODO: check me after but let's leave these to 0 first
        # Some needed constants
        ConstantOp.from_int_and_width(0, 32)

        intermediate_result = []
        # Calculating the MLP process
        for i in range(len(hidden_layers_dim)):
            # Specify the operation
            empty_tensor = EmptyOp([], hidden_output_types[i])
            if i == 0:
                # For the first layer
                result = MatmulOp(
                    [input_const.result, hidden_weight_const[i].result],
                    empty_tensor.results,
                )
            else:
                # For the other layers
                result = MatmulOp(
                    [intermediate_result[i - 1][0], hidden_weight_const[i].result],
                    empty_tensor.results,
                )

            intermediate_result.append(result.res)

        # For the output layer
        empty_tensor = EmptyOp([], output_type)
        final_result = MatmulOp(
            [intermediate_result[-1][0], output_weight_const.result],
            empty_tensor.results,
        )
        # Return both the computed result and the golden output
        ReturnOp(final_result, output_const)

    function = FuncOp.from_region("snax_main", [], res_types, func_body)
    return ModuleOp([function])


if __name__ == "__main__":
    # Debug mode
    print_data = False

    # Example usage
    batch_size = 8
    input_dim = 8
    hidden_layers_dim = [8, 16, 32, 16, 8]
    output_dim = 8

    # Print parameters
    print(f"Batch size: {batch_size}")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden layers dimensions: {hidden_layers_dim}")
    print(f"Output dimension: {output_dim}")

    # Call the cascade_matmul function
    (
        input_vals,
        hidden_weights_list,
        hidden_output_vals,
        output_weights,
        output_vals,
    ) = cascade_matmul(batch_size, input_dim, hidden_layers_dim, output_dim)

    # For debugging purposes
    # Print the results
    if print_data:
        print("Input values:")
        print(input_vals)
        print("Hidden weights:")
        for i, weights in enumerate(hidden_weights_list):
            print(f"Layer {i} weights:")
            print(weights)
        print("Intermediate values:")
        for i, intermediate in enumerate(hidden_output_vals):
            print(f"Layer {i} intermediate values:")
            print(intermediate)
        print("Output weights:")
        print(output_weights)
        print("Output values:")
        print(output_vals)

    # Get the name of the current Python script and replace its extension with .mlir
    script_name = os.path.basename(__file__)
    mlir_filename = os.path.splitext(script_name)[0] + ".mlir"

    # Generate IR and write it to the specified MLIR file
    output = StringIO()
    printer = Printer(stream=output)
    printer.print(
        mlir_cascade_matmul(
            input_vals,
            hidden_weights_list,
            hidden_output_vals,
            output_weights,
            output_vals,
        )
    )
    with open(mlir_filename, "w") as output_file:
        output_file.write(output.getvalue())
