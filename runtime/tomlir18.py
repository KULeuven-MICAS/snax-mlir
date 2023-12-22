#!/usr/bin/env python3

""" Convert an MLIR18 file to MLIR16
This program performs the following conversion steps:
    *   change operand_segment_sizes to operandSegmentSizes
    *   add double quotes to attribute names
"""

import re
import sys

if __name__ == "__main__":
    ir = sys.stdin.read()
    # add quotes
    ir = re.sub("indexing_maps", '"indexing_maps"', ir)
    # add quotes
    ir = re.sub("iterator_types", '"iterator_types"', ir)
    # add quotes and change spelling
    ir = re.sub("operand_segment_sizes", '"operandSegmentSizes"', ir)
    print(ir)
