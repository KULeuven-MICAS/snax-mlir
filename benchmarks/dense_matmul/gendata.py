# simple script to generate inputs and expected outputs for simple_matmult

import argparse

import numpy as np

from util.gendata import create_data, create_header


def create_test_data(m, n, k, add_c: bool = False):
    print(f"Creating test data with m={m}, n={n}, k={k}, add_c={add_c}")
    # Reset random seed for reproducible behavior

    np.random.seed(0)

    A_size = [m, k]
    B_size = [k, n]
    C_size = [m, n]

    # D = AxB (+ C)
    low_bound = -128
    high_bound = 127

    A = np.random.randint(low_bound, high_bound, size=A_size, dtype=np.dtype("int8"))
    B = np.random.randint(low_bound, high_bound, size=B_size, dtype=np.dtype("int8"))
    C = np.random.randint(low_bound, high_bound, size=C_size, dtype=np.dtype("int32"))

    # Make sure the product is possible!
    assert A.shape[1] == B.shape[0]

    # Compute golden output D
    D = np.matmul(A.astype(np.dtype("int32")), B.astype(np.dtype("int32")))

    if add_c:
        D += C

    sizes = {
        "M_size": A.shape[0],
        "N_size": A.shape[1],
        "K_size": B.shape[1],
    }
    variables = {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
    }

    create_header("data", sizes, variables)
    create_data("data", variables)


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Generate test data with specified parameters."
    )
    # Adding arguments
    parser.add_argument("--m", type=int, default=16, help="Value for m (default: 16)")
    parser.add_argument("--n", type=int, default=16, help="Value for n (default: 16)")
    parser.add_argument("--k", type=int, default=16, help="Value for k (default: 16)")
    parser.add_argument(
        "--add_c", type=int, default=False, help="Add C value to result"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    create_test_data(n=args.n, m=args.m, k=args.k, add_c=bool(args.add_c))
