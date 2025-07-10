import argparse
import json

import numpy as np

from util.gendata import create_header_only

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Path to the .json sample data")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the data")
    parser.add_argument("--output", type=str, default="data.h", help="Output data file")
    args = parser.parse_args()
    data = json.load(open(args.data))
    create_header_only(
        args.output,
        {
            "input": np.array(data["input"]["data"] * args.batch_size, dtype=data["input"]["dtype"]).flatten(),
            "output": np.array(data["output"]["data"] * args.batch_size, dtype=data["output"]["dtype"]).flatten(),
        },
    )
