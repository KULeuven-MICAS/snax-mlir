#!/usr/bin/env python3

import re
import sys


if __name__ == "__main__":
    ir = sys.stdin.read()
    ir = re.sub("indexing_maps", '"indexing_maps"', ir)
    ir = re.sub("iterator_types", '"iterator_types"', ir)
    ir = re.sub("operand_segment_sizes", '"operandSegmentSizes"', ir)
    print(ir)
