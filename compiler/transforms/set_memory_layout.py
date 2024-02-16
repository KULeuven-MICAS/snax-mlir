from xdsl.dialects import builtin, linalg
from xdsl.ir import MLContext
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
    to snax_qgemm by inserting layout_cast ops. In the future, the memory
    layout will be selected in a more automatic way.

    Note: currently, only snax_qgemm is supported.
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
        if library_call == "snax_qgemm":
            tsl_input_a = TiledStridedLayoutAttr(
                TiledStridedLayout(
                    [
                        TiledStride([Stride(None, None), Stride(8, 8)]),
                        TiledStride([Stride(256, None), Stride(1, 8)]),
                    ]
                )
            )

            ## tsl b has an offset of 64 to not collide with the banks of a
            tsl_input_b = TiledStridedLayoutAttr(
                TiledStridedLayout(
                    [
                        TiledStride([Stride(256, None), Stride(1, 8)]),
                        TiledStride([Stride(None, None), Stride(8, 8)]),
                    ],
                    offset=64,
                )
            )

            tsl_output = TiledStridedLayoutAttr(
                TiledStridedLayout(
                    [
                        TiledStride([Stride(256, None), Stride(4, 8)]),
                        TiledStride([Stride(None, None), Stride(32, 8)]),
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
