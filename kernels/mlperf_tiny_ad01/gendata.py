# simple script to generate inputs and expected outputs for simple_mult

import numpy as np

from util.gendata import create_data, create_header

if __name__ == "__main__":
    array_size = (8, 640)
    A = np.random.randint(0, 16, size=array_size, dtype=np.dtype("int8"))
    variables = {"A": A}
    create_header("data.h", {}, variables)
    create_data("data.c", variables)
