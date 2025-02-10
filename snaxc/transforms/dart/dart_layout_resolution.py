from dataclasses import dataclass

import numpy as np
from xdsl.context import MLContext
from xdsl.dialects import builtin, memref
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, MemRefType
from xdsl.ir import Operation
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from snaxc.dialects import dart
from snaxc.ir.dart.access_pattern import Schedule, SchedulePattern
from snaxc.ir.dart.affine_transform import AffineTransform


@dataclass
class LayoutResolution(RewritePattern):
    """
    Applies layout resolution by converting a ScheduleOp (mapping the iteration
    space to the operand index space) into an AccessOp (mapping the iteration space
    to memory), using a certain memory layout of the memref operands of the operation.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dart.ScheduleOp, rewriter: PatternRewriter):
        bounds = [x.value.data for x in op.bounds.data]
        schedule = Schedule(
            SchedulePattern(bounds, pattern.data) for pattern in op.patterns
        )

        # small function to generate a list of n zeros with the i-th element 1
        # for example n = 4, i = 1  -> [0, 1, 0, 0]
        def generate_one_list(n: int, i: int):
            return [1 if j == i else 0 for j in range(n)]

        access_patterns: list[AffineMap] = []

        # Do this for every operand:
        for operand in range(len(op.operands)):
            # Mapping from data to memory:
            assert isinstance(memref_type := op.operands[operand].type, MemRefType)

            # Mapping from data to memory:
            data_mem_map: AffineMap = memref_type.get_affine_map_in_bytes()

            # Mapping from access to data:
            access_data_map: AffineMap = schedule[operand].pattern.to_affine_map()

            # Mapping from access to memory:
            access_mem_map: AffineMap = data_mem_map.compose(access_data_map)

            # Make sure no symbols are used (not supported yet)
            if access_mem_map.num_symbols != 0:
                raise RuntimeError(
                    "Access patterns with symbols are not supported yet."
                )

            strides: list[int] = []

            for i in range(access_mem_map.num_dims):
                strides.append(
                    access_mem_map.eval(
                        generate_one_list(access_mem_map.num_dims, i), ()
                    )[0]
                )

            access_patterns.append(
                AffineTransform(np.array([strides]), np.array([0])).to_affine_map()
            )

        new_inputs: list[Operation] = [
            memref.ExtractAlignedPointerAsIndexOp.get(input) for input in op.inputs
        ]
        new_outputs = [
            memref.ExtractAlignedPointerAsIndexOp.get(output) for output in op.outputs
        ]

        new_patterns = ArrayAttr([AffineMapAttr(map) for map in access_patterns])

        access_pattern_op = dart.AccessPatternOp(
            new_inputs,
            new_outputs,
            new_patterns,
            rewriter.move_region_contents_to_new_regions(op.body),
            op.bounds,
            op.accelerator,
            op.result_types,
        )
        rewriter.replace_matched_op(
            [*new_inputs, *new_outputs, access_pattern_op], access_pattern_op.results
        )


@dataclass(frozen=True)
class DartLayoutResolutionPass(ModulePass):
    name = "dart-layout-resolution"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(LayoutResolution()).rewrite_module(op)
