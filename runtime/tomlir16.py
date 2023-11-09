#!/usr/bin/env python3

import re
import sys


if __name__ == "__main__":
    ir = sys.stdin.read()
    ir = re.sub(
        r"<({.*?})>\s*(\({[^}]*}\))", lambda x: f"{x.group(2)} {x.group(1)}", ir
    )
    ir = re.sub('"indexing_maps"', "indexing_maps", ir)
    ir = re.sub('"iterator_types"', "iterator_types", ir)
    ir = re.sub('"operandSegmentSizes"', "operand_segment_sizes", ir)
    print(ir)
