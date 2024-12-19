import os

import numpy as np
from numpy import typing as npt


def create_header(
    file_name: str, sizes: dict[str, int], variables: dict[str, npt.NDArray]
) -> None:
    header_file = f"{file_name}.h"
    if os.path.dirname(header_file):
        os.makedirs(os.path.dirname(header_file), exist_ok=True)
    with open(header_file, "w") as f:
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
    includes = [f'#include "{file_name}.h"', "", ""]
    includes = "\n".join(includes)
    c_file = f"{file_name}.c"
    variables = {i: np.reshape(j, j.size) for i, j in variables.items()}
    if os.path.dirname(c_file):
        os.makedirs(os.path.dirname(c_file), exist_ok=True)
    with open(c_file, "w") as f:
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
