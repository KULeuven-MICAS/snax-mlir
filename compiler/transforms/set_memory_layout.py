from dataclasses import dataclass

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
from xdsl.rewriter import InsertPoint
from xdsl.utils.str_enum import StrEnum

from compiler.dialects import stream
from compiler.dialects.kernel import AddOp, QMacOp, RescaleOp
from compiler.dialects.snax import LayoutCast
from compiler.dialects.tsl import TiledStridedLayoutAttr
from compiler.ir.stream import Schedule, SchedulePattern
from compiler.ir.tsl import Stride, TiledStride, TiledStridedLayout


class AddMemoryLayoutSIMD(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, linalg_op: linalg.Generic, rewriter: PatternRewriter):
        # check if operation is dispatched via library call, as set by e.g.
        # the dispatch-kernels pass
        if linalg_op.library_call is None:
            return
        else:
            library_call = linalg_op.library_call.data

        # check for library call
        if library_call == "snax_gemmx_stream":
            if not isinstance(linalg_op.body.block.first_op, RescaleOp):
                return

            shaped_operands: list[MemRefType] = [
                op.type
                for op in linalg_op.operands
                if isinstance(op.type, builtin.MemRefType)
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
                linalg_op.inputs[0], tsl_input
            )

            new_output = LayoutCast.from_type_and_target_layout(
                linalg_op.outputs[0], tsl_output
            )

            new_linalg_op = linalg.Generic(
                inputs=[new_input_a.dest],
                outputs=[new_output.dest],
                body=rewriter.move_region_contents_to_new_regions(linalg_op.regions[0]),
                indexing_maps=linalg_op.indexing_maps,
                iterator_types=linalg_op.iterator_types,
                doc=linalg_op.doc,
                library_call=linalg_op.library_call,
            )

            rewriter.insert_op_before_matched_op([new_input_a, new_output])
            rewriter.replace_op(linalg_op, new_linalg_op)

        pass


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
    def match_and_rewrite(
        self, op: linalg.Generic | stream.StreamingRegionOp, rewriter: PatternRewriter
    ):
        # check if operation is dispatched via library call, as set by e.g.
        # the dispatch-kernels pass

        if isinstance(op, linalg.Generic):
            if op.library_call is None:
                return
            else:
                library_call = op.library_call.data
        elif isinstance(op, stream.StreamingRegionOp):
            if op.accelerator is None:
                return
            else:
                library_call = op.accelerator.data

        has_add_c = False

        # check for library call
        if library_call == "snax_gemmx" or library_call == "snax_gemmx_stream":
            # only do so for qmac kernels
            if isinstance(op, linalg.Generic):
                if not isinstance(op.body.block.first_op, QMacOp):
                    return
            elif isinstance(op, stream.StreamingRegionOp):
                # TODO: this is a bit hacky, detect conv/gemm based on rank of input tensor:
                if len(op.operands[0].type.get_shape()) > 2:
                    return
                assert isinstance(
                    generic_op := op.body.block.first_op, stream.GenericOp
                )
                if not isinstance(generic_op.body.block.first_op, QMacOp):
                    return

                if isinstance(generic_op.next_op, stream.GenericOp):
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


@dataclass
class AddConvMemoryLayout(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.Generic | stream.StreamingRegionOp, rewriter: PatternRewriter
    ):
        # check if operation is dispatched via library call, as set by e.g.
        # the dispatch-kernels pass

        if isinstance(op, linalg.Generic):
            if op.library_call is None:
                return
            else:
                library_call = op.library_call.data
        elif isinstance(op, stream.StreamingRegionOp):
            if op.accelerator is None:
                return
            else:
                library_call = op.accelerator.data

        has_add_c = False

        # check for library call
        if library_call == "snax_gemmx" or library_call == "snax_gemmx_stream":
            # only do so for qmac kernels
            if isinstance(op, linalg.Generic):
                if not isinstance(op.body.block.first_op, QMacOp):
                    return
            elif isinstance(op, stream.StreamingRegionOp):
                # TODO: this is a bit hacky, detect conv/gemm based on rank of input tensor:
                if len(op.operands[0].type.get_shape()) != 4:
                    return
                assert isinstance(
                    generic_op := op.body.block.first_op, stream.GenericOp
                )
                if not isinstance(generic_op.body.block.first_op, QMacOp):
                    return

            # the layout should be as static as the memref is. no more, no less
            # get b, ox, oy, fx, fy, c, k

            shaped_operands: list[tuple[int, MemRefType]] = [
                (index, op.type)
                for index, op in enumerate(op.operands)
                if isinstance(op.type, builtin.MemRefType)
            ]

            # do not alter existing set layouts
            for _, memreftype in shaped_operands:
                if isinstance(memreftype.layout, TiledStridedLayoutAttr):
                    return

            b = shaped_operands[0][1].get_shape()[0]
            ix = shaped_operands[0][1].get_shape()[1]
            iy = shaped_operands[0][1].get_shape()[2]
            ox = shaped_operands[2][1].get_shape()[1]
            oy = shaped_operands[2][1].get_shape()[2]
            fx = shaped_operands[1][1].get_shape()[1]
            fy = shaped_operands[1][1].get_shape()[2]
            c = shaped_operands[0][1].get_shape()[3]
            k = shaped_operands[2][1].get_shape()[3]

            # lots of current limitations:
            assert b == 1
            assert ix > 0
            assert iy > 0
            assert ox > 0
            assert oy > 0
            assert fx > 0
            assert fy > 0
            assert c > 0
            assert c % 8 == 0
            assert k % 8 == 0
            assert ox % 8 == 0

            tsl_input_a = TiledStridedLayoutAttr(
                TiledStridedLayout(
                    [
                        TiledStride([Stride(ix * iy * c, 1)]),  # b
                        TiledStride([Stride(ix * 8, iy)]),  # iy
                        TiledStride([Stride(8, ix)]),  # ix
                        TiledStride([Stride(iy * ix * 8, c // 8), Stride(1, 8)]),  # c
                    ]
                )
            )

            tsl_input_b = TiledStridedLayoutAttr(
                TiledStridedLayout(
                    [
                        TiledStride(
                            [Stride(8 * fy * fx * c, k // 8), Stride(8, 8)]
                        ),  # k
                        TiledStride([Stride(64 * fy, fx)]),  # fy
                        TiledStride([Stride(64, fy)]),  # fx
                        TiledStride([Stride(64 * fy * fx, c // 8), Stride(1, 8)]),  # c
                    ]
                )
            )

            tsl_output = TiledStridedLayoutAttr(
                TiledStridedLayout(
                    [
                        TiledStride([Stride(ox * oy * k, 1)]),  # b
                        TiledStride([Stride(8, oy)]),  # oy
                        TiledStride([Stride(oy * 8, ox)]),  # ox
                        TiledStride([Stride(ox * oy * 8, k // 8), Stride(1, 8)]),  # k
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

@dataclass
class AddCyclicMemoryLayout(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stream.ScheduleOp, rewriter: PatternRewriter):

        # do not alter existing set layouts
        for operand in op.operands:
            if isinstance(operand.type, MemRefType):
                if isinstance(operand.type.layout, TiledStridedLayoutAttr):
                    return

        # recreate schedule from op
        schedule = Schedule(
            SchedulePattern(
                bounds=[x.data for x in bounds.data],
                pattern=pattern.data
            )
            for pattern, bounds in zip(op.patterns.data, op.bounds.data)
        )


        def generate_one_list(n: int, i: int):
            return [1 if j == i else 0 for j in range(n)]


        new_operands = []

        for operand, schedule_pattern in zip(op.operands, schedule):

            assert isinstance(optype := operand.type, MemRefType)

            strides = [[] for i in range(optype.get_num_dims())]
            current_stride = 1

            for i in reversed(range(schedule_pattern.num_dims)):
                result = schedule_pattern.pattern.eval(generate_one_list(schedule_pattern.num_dims, i), [])
                result_1 = [1 if x else 0 for x in result]
                if 1 in result_1:
                    dim = result_1.index(1)
                    existing_bound = 1
                    if strides[dim]:
                        existing_bound = strides[dim][0].bound
                    dim_shape = optype.get_shape()[dim] // existing_bound
                    if dim_shape % schedule_pattern.bounds[i] == 0:
                        bound = schedule_pattern.bounds[i]
                    else:
                        bound = dim_shape
                    strid_obj = Stride(current_stride, bound)
                    current_stride = current_stride * bound
                    strides[result_1.index(1)].append(strid_obj)

            layout = TiledStridedLayout([TiledStride(s) for s in strides])
            layout = layout.simplify()
            tsl = TiledStridedLayoutAttr(layout)

            # insert layout_cast ops
            new_operands.append(LayoutCast.from_type_and_target_layout(operand, tsl))

        rewriter.insert_op(new_operands, InsertPoint.before(op))

        for i, new_operand in enumerate(new_operands):
            op.operands[i] = new_operand.dest

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
        PatternRewriteWalker(AddConvMemoryLayout()).rewrite_module(op)
        PatternRewriteWalker(AddCyclicMemoryLayout()).rewrite_module(op)
