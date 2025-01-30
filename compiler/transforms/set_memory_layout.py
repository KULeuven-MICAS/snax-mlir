from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.parser import MemRefType
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.str_enum import StrEnum

from compiler.dialects import dart
from compiler.dialects.kernel import AddOp, QMacOp, RescaleOp
from compiler.dialects.snax import LayoutCast
from compiler.dialects.tsl import TiledStridedLayoutAttr
from compiler.ir.tsl import Stride, TiledStride, TiledStridedLayout


class AddMemoryLayoutSIMD(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dart.ScheduleOp, rewriter: PatternRewriter):
        # check if operation is dispatched via library call, as set by e.g.
        # the dispatch-kernels pass

        if op.accelerator is None:
            return
        else:
            library_call = op.accelerator.data

        # check for library call
        if library_call == "snax_gemmx":
            if not isinstance(op.body.block.first_op.body.block.first_op, RescaleOp):
                return

            shaped_operands: list[MemRefType] = [
                operand.type
                for operand in op.operands
                if isinstance(operand.type, builtin.MemRefType)
            ]

            m = shaped_operands[0].get_shape()[0]
            n = shaped_operands[0].get_shape()[1]

            if m == -1:
                m = None
            if n == -1:
                n = None

            tsl_input = TiledStridedLayoutAttr(
                TiledStridedLayout(
                    [
                        TiledStride(
                            [
                                Stride(
                                    256 * n // 8 if n else None, m // 8 if m else None
                                ),
                                Stride(8, 8),
                            ]
                        ),
                        TiledStride([Stride(256, n // 8 if n else None), Stride(1, 8)]),
                    ]
                )
            )

            tsl_output = TiledStridedLayoutAttr(
                TiledStridedLayout(
                    [
                        TiledStride(
                            [
                                Stride(
                                    256 * n // 8 if n else None, m // 8 if m else None
                                ),
                                Stride(8, 8),
                            ]
                        ),
                        TiledStride([Stride(256, n // 8 if n else None), Stride(1, 8)]),
                    ]
                )
            )

            # insert layout_cast ops
            new_input_a = LayoutCast.from_type_and_target_layout(
                op.inputs[0], tsl_input
            )

            new_output = LayoutCast.from_type_and_target_layout(
                op.outputs[0], tsl_output
            )

            rewriter.insert_op([new_input_a, new_output], InsertPoint.before(op))

            op.operands[0] = new_input_a.dest
            op.operands[1] = new_output.dest


class GemmLayout(StrEnum):
    cyclic = "cyclic"
    banked = "banked"


@dataclass
class AddMemoryLayout(RewritePattern):
    """
    This class represents a rewrite pattern for adding memory layout to a
    linalg operation. The implementation is very naive. It imposes a specific
    memory layout on the input and output of the linalg operation dispatched
    to snax_gemm by inserting layout_cast ops. In the future, the memory
    layout will be selected in a more automatic way.

    Note: currently, only snax_gemm is supported.
    """

    gemm_layout: GemmLayout = GemmLayout.cyclic

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dart.ScheduleOp, rewriter: PatternRewriter):
        # check if operation is dispatched via library call, as set by e.g.
        # the dispatch-kernels pass

        if op.accelerator is None:
            return
        else:
            library_call = op.accelerator.data

        has_add_c = False

        # check for library call
        if library_call == "snax_gemmx" or library_call == "snax_gemmx_stream":
            # only do so for qmac kernels
            generic_op = op.body.block.first_op
            assert isinstance(generic_op, dart.GenericOp)
            if not isinstance(generic_op.body.block.first_op, QMacOp):
                return

            if isinstance(generic_op.next_op, dart.GenericOp):
                if isinstance(generic_op.next_op.body.block.first_op, AddOp):
                    # gemm
                    has_add_c = True

            # the layout should be as static as the memref is. no more, no less
            # get m, n, k

            shaped_operands: list[tuple[int, MemRefType]] = [
                (index, op.type)
                for index, op in enumerate(op.operands)
                if isinstance(op.type, builtin.MemRefType)
            ]

            m = shaped_operands[0][1].get_shape()[0]
            n = shaped_operands[1][1].get_shape()[1]
            k = shaped_operands[0][1].get_shape()[1]

            if m == -1:
                m = None
            if n == -1:
                n = None
            if k == -1:
                k = None

            # determine tile_stride = stride between two gemm tiles
            match self.gemm_layout:
                case GemmLayout.banked:
                    tile_stride = 256
                case GemmLayout.cyclic:
                    tile_stride = 64

            tsl_input_a = TiledStridedLayoutAttr(
                TiledStridedLayout(
                    [
                        TiledStride(
                            [
                                Stride(
                                    tile_stride * k // 8 if k else None,
                                    m // 8 if m else None,
                                ),
                                Stride(8, 8),
                            ]
                        ),
                        TiledStride(
                            [Stride(tile_stride, k // 8 if k else None), Stride(1, 8)]
                        ),
                    ]
                )
            )

            ## tsl b has an offset of 64 to not collide with the banks of
            ### a (not yet - need aligned allocation for this)
            tsl_input_b = TiledStridedLayoutAttr(
                TiledStridedLayout(
                    [
                        TiledStride(
                            [Stride(tile_stride, k // 8 if k else None), Stride(1, 8)]
                        ),
                        TiledStride(
                            [
                                Stride(
                                    tile_stride * k // 8 if k else None,
                                    n // 8 if n else None,
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
                op.inputs[0], tsl_input_a
            )

            new_input_b = LayoutCast.from_type_and_target_layout(
                op.inputs[1], tsl_input_b
            )

            new_output = LayoutCast.from_type_and_target_layout(
                op.outputs[0], tsl_output
            )

            rewriter.insert_op(
                (new_input_a, new_input_b, new_output), InsertPoint.before(op)
            )

            if has_add_c:
                rewriter.insert_op(
                    new_input_c := LayoutCast.from_type_and_target_layout(
                        op.inputs[2], tsl_output
                    ),
                    InsertPoint.before(op),
                )
                op.operands[shaped_operands[0][0]] = new_input_a.dest
                op.operands[shaped_operands[1][0]] = new_input_b.dest
                op.operands[shaped_operands[2][0]] = new_input_c.dest
                op.operands[shaped_operands[3][0]] = new_output.dest
            else:
                op.operands[shaped_operands[0][0]] = new_input_a.dest
                op.operands[shaped_operands[1][0]] = new_input_b.dest
                op.operands[shaped_operands[2][0]] = new_output.dest


@dataclass(frozen=True)
class SetMemoryLayout(ModulePass):
    name = "set-memory-layout"

    gemm_layout: str = "cyclic"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            AddMemoryLayoutSIMD(), apply_recursively=False
        ).rewrite_module(op)
        PatternRewriteWalker(
            AddMemoryLayout(gemm_layout=GemmLayout(self.gemm_layout)),
            apply_recursively=False,
        ).rewrite_module(op)
