import argparse

import numpy as np

from util.gendata import create_data, create_header

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate data for snax-alu operations."
    )
    parser.add_argument(
        "--array_size", type=int, default=1024, help="Size of the arrays to generate"
    )
    args = parser.parse_args()

    low_bound = -128
    high_bound = 127
    array_size = args.array_size

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

    create_header("data.h", sizes, variables)
    create_data("data.c", variables)
