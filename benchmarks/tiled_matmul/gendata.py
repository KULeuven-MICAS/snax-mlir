# simple script to generate inputs and expected outputs for simple_matmult

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


def create_test_data(n, m, k):
    n = int(n)
    m = int(m)
    k = int(k)
    ones = False
    random_shape = False
    print(
        f"Creating test data with n={n}, m={m}, k={k}, ones={ones}, random_shape={random_shape}"
    )
    # Reset random seed for reproducible behavior

    np.random.seed(0)

    min_size = 1
    max_size = 4
    n_random = n
    m_random = m
    k_random = k

    if random_shape:
        n_random *= np.random.randint(min_size, max_size)
        m_random *= np.random.randint(min_size, max_size)
        k_random *= np.random.randint(min_size, max_size)

    A_size = [n_random, k_random]
    B_size = [k_random, m_random]

    # C = A.B
    low_bound = -128
    high_bound = 127

    if ones:
        A = np.ones(shape=A_size, dtype=np.dtype("int8"))
        B = np.ones(shape=B_size, dtype=np.dtype("int8"))
    else:
        A = np.random.randint(
            low_bound, high_bound, size=A_size, dtype=np.dtype("int8")
        )
        B = np.random.randint(
            low_bound, high_bound, size=B_size, dtype=np.dtype("int8")
        )

    # Make sure the product is possible!
    assert A.shape[1] == B.shape[0]
    C_golden = np.matmul(A.astype(np.dtype("int32")), B.astype(np.dtype("int32")))
    C = np.zeros(C_golden.shape, np.dtype("int32"))

    # Perform layout transformations before writing to memory

    # only thing necessary: transform B from row-major to column-major
    B_new_layout = np.transpose(B)

    # C are just all zeros, so layout not important
    sizes = {
        "N_size": A.shape[0],
        "K_size": A.shape[1],
        "M_size": B.shape[1],
    }
    variables = {
        "A": A,
        "B": B_new_layout,
        "C_golden": C_golden,
        "C": C,
    }

    create_header(f"data_{m}_{n}_{k}.h", sizes, variables)
    create_data(f"data_{m}_{n}_{k}.c", variables)


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Generate test data with specified parameters."
    )
    # Adding arguments
    parser.add_argument("--n", type=int, default=16, help="Value for n (default: 16)")
    parser.add_argument("--m", type=int, default=16, help="Value for m (default: 16)")
    parser.add_argument("--k", type=int, default=16, help="Value for k (default: 16)")
    parser.add_argument(
        "--ones", action="store_true", help="Use ones flag (default: False)"
    )
    parser.add_argument(
        "--random_shape",
        action="store_true",
        help="Use random_shape flag (default: False)",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    create_test_data(
        n=args.n, m=args.m, k=args.k, ones=args.ones, random_shape=args.random_shape
    )
