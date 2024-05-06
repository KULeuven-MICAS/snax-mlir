# simple script to generate inputs and expected outputs for simple_matmult
import os

import numpy as np
from numpy import typing as npt


def create_header(
    file_name: str, sizes: dict[str, int], variables: dict[str, npt.NDArray]
) -> None:
    if os.path.dirname(file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        includes = ["#include <stdint.h>", "#pragma once", ""]
        includes = "\n".join(includes)
        variables_string = [""]
        for i, j in sizes.items():
            variables_string.append(f"#define {i} {j}")
        variables_string.append("")
        for i, j in variables.items():
            variables_string.append(f"extern const {j.dtype}_t {i}[{j.size}];")
        variables_string = "\n".join(variables_string)
        f.write(includes)
        f.write(variables_string)
        f.write("\n")


def create_data(file_name: str, variables: dict[str, npt.NDArray]):
    includes = ['#include "data.h"', "", ""]
    includes = "\n".join(includes)
    variables = {i: np.reshape(j, j.size) for i, j in variables.items()}
    if os.path.dirname(file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        f.write(includes)
        for variable_name, variable_value in variables.items():
            f.write(
                f"const {variable_value.dtype}_t {variable_name}"
                + f"[{variable_value.size}] = "
                + "{\n"
            )
            variable_str = ["\t" + str(i) for i in variable_value]
            f.write(",\n".join(variable_str))
            f.write("\n};\n\n")
