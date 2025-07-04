import numpy as np
import numpy.typing as npt


def postprocessing_simd_golden_model(
    data_in: npt.NDArray[np.int_],
    input_zp_i: int,
    output_zp_i: int,
    shift_i: int | npt.NDArray[np.int_],
    max_int_i: int,
    min_int_i: int,
    double_round_i: int,
    multiplier_i: int | npt.NDArray[np.int_],
) -> npt.NDArray[np.int_]:
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
