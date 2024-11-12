# simple script to generate inputs and expected outputs for simple_mult

import numpy as np

from util.gendata import create_data, create_header

if __name__ == "__main__":
    array_size = (1, 32, 32, 3)
    np.random.seed(0)
    A = np.random.randint(-127, 128, size=array_size, dtype=np.int8)
    variables = {"A": A}
    create_header("data.h", {}, variables)
    create_data("data.c", variables)
