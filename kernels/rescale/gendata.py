import numpy as np

from util.gemmx.simd_golden_model import postprocessing_simd_golden_model
from util.gendata import create_data, create_header


def create_data_files():
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
            input_zp_i=23,
            output_zp_i=-15,
            shift_i=39,
            max_int_i=100,
            min_int_i=-110,
            double_round_i=True,
            multiplier_i=1234567890,
        )

    sizes = {"MODE": 0, "DATA_LEN": data_len}
    variables = {"A": A, "O": O, "G": G}

    create_header("data", sizes, variables)
    create_data("data", variables)
