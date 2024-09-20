from math import floor

import numpy as np

from util.gendata import create_data, create_header


# golden model for rescaling
def rescale(a, zp_in, zp_out, mult, shift, min_int, max_int, double_round):
    a = a - zp_in
    a = a * mult
    a = a / (2**shift)
    if double_round:
        a = round(a)
    else:
        a = floor(a)
    a = a + zp_out
    if a > max_int:
        a = max_int
    if a < min_int:
        a = min_int
    return a


if __name__ == "__main__":
    low_bound = -80000
    high_bound = 80000
    array_size = 16*16

    # set random seed
    np.random.seed(0)

    # G = A + B (snax-alu mode 0)
    A = np.random.randint(low_bound, high_bound, size=array_size, dtype=np.dtype("int32"))
    O = np.zeros(array_size, dtype=np.dtype("int8"))

    G = np.zeros(array_size, dtype=np.dtype("int8"))

    for i in range(O.size):
        G[i] = rescale(
            A[i], zp_in=23, zp_out=-15, mult=1234567890, shift=39, max_int=100, min_int=-110, double_round=True
        )

    sizes = {"MODE": 0, "DATA_LEN": array_size}
    variables = {"A": A, "O": O, "G": G}

    create_header("data.h", sizes, variables)
    create_data("data.c", variables)
