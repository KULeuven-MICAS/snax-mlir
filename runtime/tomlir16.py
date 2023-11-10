#!/usr/bin/env python3

""" Convert an MLIR18 file to MLIR16
This program performs the following conversion steps:
    *   take properties enclosed in <{}> and make them attributes:
        change enclosing to {} and move them to come after the 
        region of the operation
    *   change operandSegmentSizes to operand_segment_sizes
    *   remove double quotes from attribute names 
"""

import re
import sys


if __name__ == "__main__":
    ir = sys.stdin.read()

    # swap attributes and region from place, and remove <> from attributes
    ir = re.sub(
        r"<({.*?})>\s*(\({[^}]*}\))", lambda x: f"{x.group(2)} {x.group(1)}", ir
    )
    # remove quotes
    ir = re.sub('"indexing_maps"', "indexing_maps", ir)
    # remove quotes
    ir = re.sub('"iterator_types"', "iterator_types", ir)
    # remove quotes and change spelling
    ir = re.sub('"operandSegmentSizes"', "operand_segment_sizes", ir)
    print(ir)
