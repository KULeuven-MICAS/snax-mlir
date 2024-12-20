import numpy as np

from util.gendata import create_data, create_header


def create_data_files():
    # Reset random seed for reproducible behavior

    np.random.seed(0)

    n = 16
    m = 16
    k = 16

    A_size = [m, k]
    B_size = [k, n]

    # D = A.B + C
    low_bound = -128
    high_bound = 127
    A = np.random.randint(low_bound, high_bound, size=A_size, dtype=np.dtype("int8"))
    B = np.random.randint(low_bound, high_bound, size=B_size, dtype=np.dtype("int8"))

    C = np.random.randint(low_bound, high_bound, size=B_size, dtype=np.dtype("int32"))

    # Make sure the product is possible!
    assert A.shape[1] == B.shape[0]
    D_golden = np.matmul(A.astype(np.dtype("int32")), B.astype(np.dtype("int32"))) + C

    sizes = {
        "N_size": A.shape[0],
        "K_size": A.shape[1],
        "M_size": B.shape[1],
    }
    variables = {
        "A": A,
        "B": B,
        "C": C,
        "D_golden": D_golden,
    }

    create_header("data.h", sizes, variables)
    create_data("data.c", variables)
