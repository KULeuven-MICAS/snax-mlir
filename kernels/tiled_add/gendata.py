import argparse
import os

import numpy as np
import numpy.typing as npt


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
    includes = [f'#include "{file_name[:-2]}.h"', "", ""]
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


def generate_data(array_size):
    low_bound = -128
    high_bound = 127

    # snax-alu design-time spatial parallelism
    spatial_par = 4
    loop_iter = array_size // spatial_par

    # set random seed
    np.random.seed(0)

    # G = A + B (snax-alu mode 0)
    A = np.random.randint(
        low_bound, high_bound, size=array_size, dtype=np.dtype("int64")
    )
    B = np.random.randint(
        low_bound, high_bound, size=array_size, dtype=np.dtype("int64")
    )
    O = np.zeros(array_size, dtype=np.dtype("int64"))
    G = A + B

    sizes = {"MODE": 0, "DATA_LEN": array_size, "LOOP_ITER": loop_iter}
    variables = {"A": A, "B": B, "O": O, "G": G}

    create_header(f"data_{array_size}.h", sizes, variables)
    create_data(f"data_{array_size}.c", variables)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate data for snax-alu operations."
    )
    parser.add_argument(
        "--array_size", type=int, default=1024, help="Size of the arrays to generate"
    )
    args = parser.parse_args()
    generate_data(args.array_size)
