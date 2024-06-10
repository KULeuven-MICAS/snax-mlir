# simple script to generate inputs and expected outputs for simple_mult

import numpy as np

from util.gendata import create_data, create_header

if __name__ == "__main__":
    # Reset random seed for reproducible behavior
    low_bound = -128
    high_bound = 127
    array_size = 10
    np.random.seed(0)
    # G = A*B
    A = np.random.randint(
        low_bound, high_bound, size=array_size, dtype=np.dtype("int32")
    )
    B = np.random.randint(
        low_bound, high_bound, size=array_size, dtype=np.dtype("int32")
    )
    D = np.zeros(array_size, dtype=np.dtype("int32"))
    G = A * B
    sizes = {"N": array_size}
    variables = {"A": A, "B": B, "D": D, "G": G}
    create_header("data.h", sizes, variables)
    create_data("data.c", variables)
