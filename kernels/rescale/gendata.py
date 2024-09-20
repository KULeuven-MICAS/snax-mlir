from math import floor

import numpy as np

from util.gendata import create_data, create_header


def postprocessing_simd_golden_model(
    data_in,
    input_zp_i,
    output_zp_i,
    shift_i,
    max_int_i,
    min_int_i,
    double_round_i,
    multiplier_i,
):
    # Step 1: Subtract input zero point
    var = data_in - input_zp_i

    # Step 2: Multiply with the multiplier avoiding overflow
    var = np.int64(var) * np.int64(multiplier_i)

    # Step 3: Right shift
    var = np.int32(var >> (shift_i - 1))

    # Step 4: Apply double rounding if necessary
    if double_round_i:
        var = np.where(var >= 0, var + 1, var - 1)

    # Step 5: Final right shift
    var = var >> 1

    # Step 6: Add output zero point
    var = var + output_zp_i

    # Step 7: Clip the values to be within min and max integer range
    var = np.clip(var, min_int_i, max_int_i)

    return var


if __name__ == "__main__":
    low_bound = -80000
    high_bound = 80000
    data_len = 16

    # set random seed
    np.random.seed(0)

    # G = A + B (snax-alu mode 0)
    A = np.random.randint(low_bound, high_bound, size=data_len * data_len, dtype=np.dtype("int32"))
    O = np.zeros(data_len * data_len, dtype=np.dtype("int8"))

    G = np.zeros(data_len * data_len, dtype=np.dtype("int8"))

    for i in range(O.size):
        G[i] = postprocessing_simd_golden_model(
            A[i],
            input_zp_i= 23,
            output_zp_i=-15,
            shift_i=39,
            max_int_i=100,
            min_int_i=-110,
            double_round_i=True,
            multiplier_i=1234567890,
        )

    sizes = {"MODE": 0, "DATA_LEN": data_len}
    variables = {"A": A, "O": O, "G": G}

    create_header("data.h", sizes, variables)
    create_data("data.c", variables)
