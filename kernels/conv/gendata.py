# simple script to generate inputs and expected outputs for simple_matmult

import numpy as np
import torch.nn as nn

from util.gendata import create_data, create_header

if __name__ == "__main__":
    # Reset random seed for reproducible behavior

    np.random.seed(0)

    I_size = [1, 18, 18, 16]
    W_size = [16, 3, 3, 16]
    O_size = [1, 16, 16, 16]

    # D = A.B + C
    low_bound = -128
    high_bound = 127
    I = np.random.randint(low_bound, high_bound, size=I_size, dtype=np.dtype("int8"))
    W = np.random.randint(low_bound, high_bound, size=W_size, dtype=np.dtype("int8"))

    # TODO:: calculate output ass well
    O = np.zeros(shape=O_size, dtype=np.int32)
    O_golden = np.zeros(shape=O_size, dtype=np.int32)

    variables = {
        "I": I,
        "W": W,
        "O": O,
        "O_golden": O_golden,
    }

    create_header("data.h", {}, variables)
    create_data("data.c", variables)
