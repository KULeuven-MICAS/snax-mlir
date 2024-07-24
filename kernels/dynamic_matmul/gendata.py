# simple script to generate inputs and expected outputs for simple_matmult

import numpy as np

from util.gendata import create_data, create_header

if __name__ == "__main__":
    # Reset random seed for reproducible behavior

    np.random.seed(0)

    min_size = 1
    max_size = 4
    n_random = 8 * np.random.randint(min_size, max_size)
    m_random = 8 * np.random.randint(min_size, max_size)
    k_random = 8 * np.random.randint(min_size, max_size)
    A_size = [n_random, k_random]
    B_size = [k_random, m_random]

    # C = A.B
    low_bound = -128
    high_bound = 127
    A = np.random.randint(low_bound, high_bound, size=A_size, dtype=np.dtype("int8"))
    B = np.random.randint(low_bound, high_bound, size=B_size, dtype=np.dtype("int8"))

    # A = np.ones(shape=A_size, dtype=np.dtype("int8"))
    # B = np.ones(shape=B_size, dtype=np.dtype("int8"))

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

    create_header("data.h", sizes, variables)
    create_data("data.c", variables)
