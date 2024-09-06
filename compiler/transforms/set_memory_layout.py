from xdsl.context import MLContext
from xdsl.dialects import builtin, linalg
from xdsl.parser import MemRefType
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.dialects.snax import LayoutCast
from compiler.dialects.tsl import TiledStridedLayoutAttr
from compiler.ir.tsl import Stride, TiledStride, TiledStridedLayout


class AddMemoryLayout(RewritePattern):
    """
    This class represents a rewrite pattern for adding memory layout to a
    linalg operation. The implementation is very naive. It imposes a specific
    memory layout on the input and output of the linalg operation dispatched
    to snax_gemm by inserting layout_cast ops. In the future, the memory
    layout will be selected in a more automatic way.

    Note: currently, only snax_gemm is supported.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, linalg_op: linalg.Generic, rewriter: PatternRewriter):
        # check if operation is dispatched via library call, as set by e.g.
        # the dispatch-kernels pass
        if linalg_op.library_call is None:
            return
        else:
            library_call = linalg_op.library_call.data

        # check for library call
        if library_call == "snax_gemm" or library_call == "snax_gemm_stream":
            # the layout should be as static as the memref is. no more, no less
            # get m, n, k

            shaped_operands: list[MemRefType] = [
                op.type
                for op in linalg_op.operands
                if isinstance(op.type, builtin.MemRefType)
            ]

            m = shaped_operands[0].get_shape()[0]
            n = shaped_operands[1].get_shape()[1]
            k = shaped_operands[0].get_shape()[1]

            if m == -1:
                m = None
            if n == -1:
                n = None
            if k == -1:
                k = None

            tsl_input_a = TiledStridedLayoutAttr(
                TiledStridedLayout(
                    [
                        TiledStride(
                            [
                                Stride(
                                    256 * k // 8 if k else None, m // 8 if m else None
                                ),
                                Stride(8, 8),
                            ]
                        ),
                        TiledStride([Stride(256, k // 8 if k else None), Stride(1, 8)]),
                    ]
                )
            )

            ## tsl b has an offset of 64 to not collide with the banks of
            ### a (not yet - need aligned allocation for this)
            tsl_input_b = TiledStridedLayoutAttr(
                TiledStridedLayout(
                    [
                        TiledStride([Stride(256, k // 8 if k else None), Stride(1, 8)]),
                        TiledStride(
                            [
                                Stride(
                                    256 * k // 8 if k else None, n // 8 if n else None
                                ),
                                Stride(8, 8),
                            ]
                        ),
                    ],
                    # offset=64,
                )
            )

            tsl_output = TiledStridedLayoutAttr(
                TiledStridedLayout(
                    [
                        TiledStride(
                            [
                                Stride(
                                    64 * n // 8 if n else None, m // 8 if m else None
                                ),
                                Stride(8, 8),
                            ]
                        ),
                        TiledStride([Stride(64, n // 8 if n else None), Stride(1, 8)]),
                    ]
                )
            )

            # insert layout_cast ops
            new_input_a = LayoutCast.from_type_and_target_layout(
                linalg_op.inputs[0], tsl_input_a
            )

            new_input_b = LayoutCast.from_type_and_target_layout(
                linalg_op.inputs[1], tsl_input_b
            )

            new_output = LayoutCast.from_type_and_target_layout(
                linalg_op.outputs[0], tsl_output
            )

            new_linalg_op = linalg.Generic(
                inputs=[new_input_a, new_input_b, *linalg_op.inputs[2:]],
                outputs=[new_output],
                body=rewriter.move_region_contents_to_new_regions(linalg_op.regions[0]),
                indexing_maps=linalg_op.indexing_maps,
                iterator_types=linalg_op.iterator_types,
                doc=linalg_op.doc,
                library_call=linalg_op.library_call,
            )

            rewriter.insert_op_before_matched_op([new_input_a, new_input_b, new_output])
            rewriter.replace_op(linalg_op, new_linalg_op)

        pass


class SetMemoryLayout(ModulePass):
    name = "set-memory-layout"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(AddMemoryLayout(), apply_recursively=False).rewrite_module(
            op
        )
