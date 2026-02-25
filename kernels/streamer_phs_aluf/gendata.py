import numpy as np

from util.gendata import create_data, create_header


def create_data_files():
    # Reset random seed for reproducible behavior
    low_bound = 0.0
    high_bound = 100.0
    array_size = 16

    # snax-alu design-time spatial parallelism
    spatial_par = 4
    loop_iter = array_size // spatial_par

    # set random seed
    np.random.seed(0)

    # G = A + B (snax-alu mode 0)
    # Switched to uniform for float generation and dtype to float64
    A = np.random.uniform(low_bound, high_bound, size=array_size).astype(np.float64)
    B = np.random.uniform(low_bound, high_bound, size=array_size).astype(np.float64)
    O = np.zeros(array_size, dtype=np.float64)
    G = A + B

    sizes = {"MODE": 0, "DATA_LEN": array_size, "LOOP_ITER": loop_iter}
    variables = {"A": A, "B": B, "O": O, "G": G}

    create_header("data", sizes, variables)
    create_data("data", variables)


create_data_files()
