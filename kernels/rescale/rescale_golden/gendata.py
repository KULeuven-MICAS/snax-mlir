import argparse

import numpy as np

from util.gendata import create_header_only

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitwidth", type=int, default=8, help="Bitwidth of the input data")
    parser.add_argument("--batch_size", type=int, help="Batch size for the data")
    parser.add_argument("--output", type=str, default="data.h", help="Output data file")
    args = parser.parse_args()
    # Generate random input data based on bitwidth and batch_size
    dtype = np.int8 if args.bitwidth <= 8 else np.int16 if args.bitwidth <= 16 else np.int32
    max_val = 2**args.bitwidth - 2
    input_data = np.random.randint(-(10**7), 10**7, size=args.batch_size, dtype=dtype)

    create_header_only(
        args.output,
        {
            "input": input_data,
        },
    )
