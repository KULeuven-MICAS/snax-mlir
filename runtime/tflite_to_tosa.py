#!/usr/bin/env python3

""" 
"""

import sys
import tensorflow as tf


if __name__ == "__main__":
    # Check the number of arguments
    if len(sys.argv) != 5:
        print("Usage: python tflite_to_tosa.py -c input_source -o output_file")
        sys.exit(1)

    # Extract the arguments
    script_name, source_flag, input_source, output_flag, output_file = sys.argv

    # Check if the flags are correct
    if source_flag != "-c" or output_flag != "-o":
        print("Invalid flags. Use -c for input source and -o for output file.")
        sys.exit(1)

    # Your code here to use input_source and output_file

    tf.mlir.experimental.tflite_to_tosa_bytecode(
        input_source,
        output_file,
        use_external_constant=False,
        ordered_input_arrays=None,
        ordered_output_arrays=None,
    )
