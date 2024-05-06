# simple script to generate inputs and expected outputs for simple_mult

import numpy as np

from util.gendata import create_data, create_header

if __name__ == "__main__":
    array_size = 10
    A = np.linspace(1, array_size, array_size, dtype=np.int32)
    sizes = {"N": array_size}
    variables = {"A": A}
    create_header("data.h", sizes, variables)
    create_data("data.c", variables)
