import os
from math import sqrt

import numpy as np
from numpy import typing as npt
from testcases import testcases
from xdsl.builder import ImplicitBuilder
from xdsl.dialects.builtin import i32
from xdsl.dialects.func import FuncOp, Return
from xdsl.dialects.memref import CopyOp, MemRefType

from compiler.dialects.tsl import TiledStridedLayoutAttr


def create_header(file_name: str, size: int, variables: dict[str, npt.NDArray]) -> None:
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        includes = [
            "#include <stdint.h>",
            "#pragma once",
            "",
            f"#define N {size}",
            f"#define N_sqrt {sqrt(size)}",
            "",
        ]
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
        B = np.reshape(B, reshape_var(sqrt(array_size_var)))
    for sv in swapaxes_var:
        B = np.swapaxes(B, sv[0], sv[1])
    B = B.flatten()
    return {"A": A, "B": B}


def generate_mlir(tslsource, tsldest, shape="?x?"):
    memref_src = MemRefType(i32, shape, TiledStridedLayoutAttr(tslsource), 0)
    memref_dst = MemRefType(i32, shape, TiledStridedLayoutAttr(tsldest), 0)
    func_op = FuncOp(
        "transform_copy", ((memref_src, memref_dst), ()), visibility="public"
    )
    with ImplicitBuilder(func_op.body):
        CopyOp(func_op.args[0], func_op.args[1])
        Return()
    return str(func_op)


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
                    "shape": testcase["shape"],
                    "tslsrc": testcase["tslsrc"],
                    "tsldst": testcase["tsldst"],
                    "data": data_generator(
                        testcase["reshape_var"],
                        testcase["swapaxis_var"],
                        array_size,
                    ),
                }
            )
    return result


if __name__ == "__main__":
    testcases = parse_testcases(testcases)
    for testcase in testcases:
        size_sqrt = int(sqrt(testcase["size"]))
        # data = testcase["generator"](testcase["size"])
        create_header(testcase["name"] + "/data.h", testcase["size"], testcase["data"])
        create_data(testcase["name"] + "/data.c", testcase["size"], testcase["data"])
        # Generate MLIR and write it to a file
        mlir = generate_mlir(
            testcase["tslsrc"](size_sqrt),
            testcase["tsldst"](size_sqrt),
            testcase["shape"](size_sqrt),
        )
        with open(testcase["name"] + ".preproc.mlir", "w") as file:
            file.write(mlir)
