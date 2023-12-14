# simple script to generate inputs and expected outputs for simple_matmult
import numpy as np
from numpy import typing as npt
from typing import Dict


def create_header(
    file_name: str, sizes: Dict[str, int], variables: Dict[str, npt.NDArray]
) -> None:
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


def create_data(file_name: str, variables: Dict[str, npt.NDArray]):
    includes = ['#include "data.h"', "", ""]
    includes = "\n".join(includes)
    variables = {i: np.reshape(j, j.size) for i, j in variables.items()}
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


if __name__ == "__main__":
    # Reset random seed for reproducible behavior
    low_bound = -128
    high_bound = 127
    A_size = [16, 16]
    B_size = [16, 16]
    np.random.seed(0)
    # G = A*B
    A = np.random.randint(low_bound, high_bound, size=A_size, dtype=np.dtype("int8"))
    # convert from row-major to block row-major
    A = np.reshape(A, [2, 8, 2, 8])
    # convert to [2,2,8,8]
    A = np.swapaxes(A, 1, 2)
    B = np.random.randint(low_bound, high_bound, size=B_size, dtype=np.dtype("int8"))
    # convert from column-major to block column-major
    B = np.reshape(B, [2, 8, 2, 8])
    # convert to [2,2,8,8]
    B = np.swapaxes(B, 1, 2)
    C_golden = np.matmul(A.astype(np.dtype("int32")), B.astype(np.dtype("int32")))
    C_golden = np.reshape(C_golden, [2, 8, 2, 8])
    C_golden = np.swapaxes(C_golden, 1, 2)
    C = np.zeros(C_golden.shape, np.dtype("int32"))
    variables = {"A": A, "B": B, "C_golden": C_golden, "C": C}
    assert A.shape[1] == B.shape[0]
    sizes = {"N_size": A.shape[0], "K_size": A.shape[1], "M_size": B.shape[1]}
    create_header("data.h", sizes, variables)
    create_data("data.c", variables)
