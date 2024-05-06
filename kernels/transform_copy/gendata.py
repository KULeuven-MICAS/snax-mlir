# simple script to generate inputs and expected outputs for simple_mult
from math import sqrt

import numpy as np

from util.gendata import create_data, create_header

if __name__ == "__main__":
    array_size = 64
    A = np.linspace(1, array_size, array_size, dtype=np.int32)
    B = np.reshape(A, [2, 4, 2, 4])
    B = np.swapaxes(B, 1, 2)
    B = B.flatten()
    sizes = {"N": array_size, "N_sqrt": sqrt(array_size)}
    variables = {"A": A, "B": B}
    create_header("transform_copy/data.h", sizes, variables)
    create_data("transform_copy/data.c", variables)
    create_header("transform_from_none/data.h", sizes, variables)
    create_data("transform_from_none/data.c", variables)
    create_header("transform_from_strided/data.h", sizes, variables)
    create_data("transform_from_strided/data.c", variables)
