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
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        f.write(includes)
        for variable_name, variable_value in variables.items():
            f.write(f"const int32_t {variable_name}[N] = " + "{\n")
            variable_str = ["\t" + str(i) for i in variable_value]
            f.write(",\n".join(variable_str))
            f.write("\n};\n\n")


def data_generator(reshape_var, swapaxes_var, array_size_var):
    A = np.linspace(1, array_size_var, array_size_var, dtype=np.int32)
    B = np.copy(A)
    if reshape_var is not None:
        B = np.reshape(B, reshape_var)
    if swapaxes_var is not None:
        B = np.swapaxes(B, swapaxes_var[0], swapaxes_var[1])
    B = B.flatten()
    return {"A": A, "B": B}


def generate_mlir(tslsource, tsldest):
    return f"""builtin.module {{
    func.func public @transform_copy(
        %arg0 : memref<?x?xi32, #tsl.tsl<{tslsource}>, 0 : i32>,
        %arg1 : memref<?x?xi32, #tsl.tsl<{tsldest}>, 1 : i32>) {{
        "memref.copy"(%arg0, %arg1) : (
            memref<?x?xi32, #tsl.tsl<{tslsource}>, 0 : i32>,
            memref<?x?xi32, #tsl.tsl<{tsldest}>, 1 : i32>) -> ()
        func.return
    }}
}}"""


def parse_testcases(testcases):
    # generates a separate test case for each array size
    # in the original testcases argument
    result = []
    for testcase in testcases:
        for array_size in testcase["array_sizes"]:
            result.append(
                {
                    "name": testcase["name"] + "_" + str(array_size) + "_gen",
                    "size": array_size,
                    "tslsrc": testcase["tslsrc"],
                    "tsldst": testcase["tsldst"],
                    "generator": testcase["generator"],
                }
            )
    return result


testcases = [
    {
        "name": "equal_layout",
        "array_sizes": [8 * 8],
        "tsldst": "[?, 4] -> (16, 4), [?, 4] -> (?, ?)",
        "tslsrc": "[?, 4] -> (16, 4), [?, 4] -> (?, ?)",
        "generator": lambda size: data_generator(None, None, size),
    }
]

if __name__ == "__main__":
    testcases = parse_testcases(testcases)
    for testcase in testcases:
        data = testcase["generator"](testcase["size"])
        create_header(testcase["name"] + "/data.h", testcase["size"], data)
        create_data(testcase["name"] + "/data.c", testcase["size"], data)
        # Generate MLIR and write it to a file
        mlir = generate_mlir(testcase["tslsrc"], testcase["tsldst"])
        with open(testcase["name"] + ".preproc.mlir", "w") as file:
            file.write(mlir)
