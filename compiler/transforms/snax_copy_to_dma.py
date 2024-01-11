from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.dialects.arith import Addi, Constant, Muli
from xdsl.dialects.builtin import IndexType, IntegerType, NoneAttr
from xdsl.dialects.memref import CopyOp, Dim, ExtractAlignedPointerAsIndexOp, MemRefType
from xdsl.ir import Block, MLContext, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable

from compiler.dialects.tsl import TiledStridedLayoutAttr
from compiler.ir.tsl.stride import Stride


class Match1DDMA(RewritePattern):
    """
    Looks for simple dense memref copy operations and insert a snitch 1d dma call
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CopyOp, rewriter: PatternRewriter):
        # only works on memrefs with nonetype layouts and equal shape and element type
        if not isinstance(op.source.type, MemRefType):
            return
        if not isinstance(op.destination.type, MemRefType):
            return
        if not isinstance(op.source.type.layout, NoneAttr):
            return
        if not isinstance(op.destination.type.layout, NoneAttr):
            return
        if not op.destination.type.get_shape() == op.source.type.get_shape():
            return
        if (
            not op.destination.type.get_element_type()
            == op.source.type.get_element_type()
        ):
            return
        if not isinstance(op.source.type.get_element_type(), IntegerType):
            return

        # step 1: extract size information to calculate total size
        ops_to_insert = []
        total_size_op = None
        for dim in range(op.source.type.get_num_dims()):
            const_op = Constant.from_int_and_width(dim, IndexType())
            ops_to_insert.append(const_op)
            dim_op = Dim.from_source_and_index(op.source, const_op.result)
            ops_to_insert.append(dim_op)
            if total_size_op is None:
                total_size_op = dim_op
            else:
                total_size_op = Muli(total_size_op.result, dim_op.result, IndexType())
                ops_to_insert.append(const_op)

        # step 2: calculate element size to get total size in bytes
        element_type = op.source.type.get_element_type()
        element_size = IntegerType.get_bit_width(element_type) // 8
        total_size_op = Muli(
            total_size_op.result,
            Constant.from_int_and_width(element_size, IndexType()).result,
            IndexType(),
        )
        ops_to_insert.append(total_size_op)

        # step 3: extract source and destination pointers
        source_ptr_op = ExtractAlignedPointerAsIndexOp.get(op.source)
        dest_ptr_op = ExtractAlignedPointerAsIndexOp.get(op.destination)
        ops_to_insert.append(source_ptr_op)
        ops_to_insert.append(dest_ptr_op)

        # step 4: create function call
        func_call = func.Call(
            "snax_dma_1d_transfer",
            [
                source_ptr_op.aligned_pointer,
                dest_ptr_op.aligned_pointer,
                total_size_op.result,
            ],
            [],
        )

        # step 5: insert ops and replace op
        rewriter.insert_op_before_matched_op(ops_to_insert)
        rewriter.replace_op(op, func_call)


class TransformDMA(RewritePattern):
    """Look for memref copy operations with TSL layout and insert snitch DMA calls"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CopyOp, rewriter: PatternRewriter):
        # only works on memrefs with tsl layouts and equal shape and element type
        if not isinstance(op.source.type, MemRefType):
            return
        if not isinstance(op.destination.type, MemRefType):
            return
        if not isinstance(op.source.type.layout, TiledStridedLayoutAttr):
            return
        if not isinstance(op.destination.type.layout, TiledStridedLayoutAttr):
            return
        if not op.destination.type.get_shape() == op.source.type.get_shape():
            return
        if (
            not op.destination.type.get_element_type()
            == op.source.type.get_element_type()
        ):
            return
        if not isinstance(op.source.type.get_element_type(), IntegerType):
            return

        ops_to_insert = []

        # step 1: extract base addresses
        pointer_src = ExtractAlignedPointerAsIndexOp.get(op.source)
        pointer_dst = ExtractAlignedPointerAsIndexOp.get(op.destination)
        ops_to_insert.append(pointer_src)
        ops_to_insert.append(pointer_dst)

        # step 2: find largest common contiguous block, to be used for dma transfers
        tsl_source = op.source.type.layout.data
        tsl_dest = op.destination.type.layout.data

        lcb = tsl_source.largest_common_contiguous_block(tsl_dest)

        # step 3: generate a sorted list of remaing strides;
        # all strides excluded from the contiguous block must be generated
        # using for loops / multi-dimensional dma transfers
        remaining_strides = [stride for stride in tsl_source if stride[2] not in lcb]
        # sort the remaining strides by their bound, largest first
        remaining_strides = sorted(
            remaining_strides, key=lambda stride: stride[2].bound, reverse=True
        )
        # map the list to [(src_stride, dest_stride), ...]
        remaining_strides = [
            (stride[2], tsl_dest.get_stride(stride[0], stride[1]))
            for stride in remaining_strides
        ]

        # step 4: generate variables for 2D transfer

        if len(remaining_strides) == 0:
            # is 1d dma transfer
            dma_loop = (Stride(1, 1), Stride(1, 1))
        else:
            dma_loop = remaining_strides.pop(0)

        dma_size = Constant.from_int_and_width(lcb[-1].bound, IndexType())
        dma_stride_src = Constant.from_int_and_width(dma_loop[0].stride, IndexType())
        dma_stride_dst = Constant.from_int_and_width(dma_loop[1].stride, IndexType())
        dma_stride_bound = Constant.from_int_and_width(dma_loop[0].bound, IndexType())
        ops_to_insert.extend(
            [dma_size, dma_stride_src, dma_stride_dst, dma_stride_bound]
        )

        # step 5: if there are no remaining strides, insert simple 2D dma transfer
        if len(remaining_strides) == 0:
            func_call = func.Call(
                "snax_dma_2d_transfer",
                [
                    pointer_src.aligned_pointer,
                    pointer_dst.aligned_pointer,
                    dma_size.result,
                    dma_stride_src.result,
                    dma_stride_dst.result,
                    dma_stride_bound.result,
                ],
                [],
            )
            rewriter.insert_op_before_matched_op(ops_to_insert)
            rewriter.replace_op(op, func_call)
            return

        # step 6: else, generate nested for loops for remaining strides
        """
        We want to generate a loop nest like this:

        for (i = 0; i < dim_0; i++) {
            for (j = 0; j < dim_1; j++) {
                snrt_2d_dma_transfer(
                    source = base_src + i * stride_1 + j * stride_2,
                    destination = base_dest + i * stride_1 + j * stride_2,
                    size = #from lcb
                    stride_src = stride_0,
                    stride_dest = stride_0
                )
            }
        }
        """

        # step 6.1: create the list of loop bounds
        lower = arith.Constant.from_int_and_width(0, builtin.IndexType())
        step = arith.Constant.from_int_and_width(1, builtin.IndexType())
        upper = [
            arith.Constant.from_int_and_width(stride[0].bound, builtin.IndexType())
            for stride in remaining_strides
        ]

        ops_to_insert.extend([lower, step, *upper])

        # step 6.2: create nested for loop (looping from inner to outer)
        # most inner for loop has empty region
        empty_region = Region(Block([scf.Yield()], arg_types=(IndexType(),)))
        for_loop = scf.For(lower, upper[0], step, [], empty_region)

        for i in range(len(remaining_strides) - 1):
            # other for loops have a region with the previous for loop as body
            region = Region(Block([for_loop, scf.Yield()], arg_types=(IndexType(),)))
            for_loop = scf.For(lower, upper[i + 1], step, [], region)

        # save outermost for loop to insert at the end
        outermost_for_loop = for_loop

        # step 6.3: insert indexing operations in for loop nest

        pointer_src = pointer_src
        pointer_dst = pointer_dst
        for i in range(len(remaining_strides)):
            next_for_op = for_loop.body.block.first_op
            # insert the ops in the for loop body
            ops_to_insert_for_loop = []

            # source indexing operations:
            stride_src = Constant.from_int_and_width(
                remaining_strides[i][0].stride, IndexType()
            )
            increment_src = Muli(for_loop.body.block.args[0], stride_src, IndexType())
            pointer_src = Addi(pointer_src, increment_src, IndexType())
            ops_to_insert_for_loop.extend([stride_src, increment_src, pointer_src])

            # destination indexing operations:
            stride_dst = Constant.from_int_and_width(
                remaining_strides[i][1].stride, IndexType()
            )
            increment_dst = Muli(for_loop.body.block.args[0], stride_dst, IndexType())
            pointer_dst = Addi(pointer_dst, increment_dst, IndexType())
            ops_to_insert_for_loop.extend([stride_dst, increment_dst, pointer_dst])

            # insert the ops in the for loop body
            for_loop.body.block.insert_ops_before(
                ops_to_insert_for_loop, for_loop.body.block.first_op
            )

            # if this is most inner for loop, also insert dma function call
            if isinstance(next_for_op, scf.Yield):
                func_call = func.Call(
                    "snax_dma_2d_transfer",
                    [
                        pointer_src,
                        pointer_dst,
                        dma_size,
                        dma_stride_src,
                        dma_stride_dst,
                        dma_stride_bound,
                    ],
                    [],
                )
                for_loop.body.block.insert_op_before(
                    func_call, for_loop.body.block.last_op
                )

            # else continue with next for loop
            else:
                for_loop = next_for_op

        rewriter.insert_op_before_matched_op(ops_to_insert)
        rewriter.replace_op(op, outermost_for_loop)


class SNAXCopyToDMA(ModulePass):
    """
    This pass translates memref copies to snitch DMA calls.
    """

    name = "snax-copy-to-dma"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        contains_copies = any(
            isinstance(op_in_module, memref.CopyOp) for op_in_module in op.walk()
        )

        if contains_copies:
            PatternRewriteWalker(Match1DDMA()).rewrite_module(op)
            PatternRewriteWalker(TransformDMA()).rewrite_module(op)
            func_decl = func.FuncOp.external(
                "snax_dma_1d_transfer", 3 * [builtin.IndexType()], []
            )
            SymbolTable.insert_or_update(op, func_decl)
            func_decl = func.FuncOp.external(
                "snax_dma_2d_transfer", 6 * [builtin.IndexType()], []
            )
            SymbolTable.insert_or_update(op, func_decl)
