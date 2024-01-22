# simple script to generate inputs and expected outputs for simple_mult

import os

import numpy as np
from numpy import typing as npt


def create_header(file_name: str, size: int, variables: dict[str, npt.NDArray]) -> None:
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        includes = ["#include <stdint.h>", "#pragma once", "", f"#define N {size}", ""]
        includes = "\n".join(includes)
        variable_names = list(variables.keys())
        variables_string = [f"extern const int32_t {i}[N];" for i in variable_names]
        variables_string = "\n".join(variables_string)
        f.write(includes)
        f.write(variables_string)
        f.write("\n")


def create_data(file_name: str, size: int, variables: dict[str, npt.NDArray]):
    includes = ['#include "data.h"', "", ""]
    includes = "\n".join(includes)
    # create folder if does not exist
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        f.write(includes)
        for variable_name, variable_value in variables.items():
            f.write(f"const int32_t {variable_name}[N] = " + "{\n")
            variable_str = ["\t" + str(i) for i in variable_value]
            f.write(",\n".join(variable_str))
            f.write("\n};\n\n")


if __name__ == "__main__":
    array_size = 64
    A = np.linspace(1, array_size, array_size, dtype=np.int32)
    B = np.reshape(A, [2, 4, 2, 4])
    B = np.swapaxes(B, 1, 2)
    B = B.flatten()
    variables = {"A": A, "B": B}
    create_header("transform_copy/data.h", array_size, variables)
    create_data("transform_copy/data.c", array_size, variables)
