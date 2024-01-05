from xdsl.dialects import arith, builtin, func, memref, scf
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


class InsertFunctionCalls(RewritePattern):
    """
    Looks for memref copy operations and insert a snitch 1d dma call
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.CopyOp, rewriter: PatternRewriter):
        if not isinstance(op.source.type, memref.MemRefType) and not isinstance(
            op.destination.type, memref.MemRefType
        ):
            return

        # Case 1: 1D memrefs without layout information
        # resolve by simple call to snrt_1d_dma_transfer
        if (
            op.source.type.get_num_dims() == 1
            and op.destination.type.get_num_dims() == 1
            and op.source.type.layout is builtin.NoneAttr
            and op.destination.type.layout is builtin.NoneAttr
        ):
            # Extract size information
            zero_const = arith.Constant.from_int_and_width(0, builtin.IndexType())
            dim_op = memref.Dim.from_source_and_index(op.source, zero_const.result)

            # Extract source and destination pointers
            source_ptr_op = memref.ExtractAlignedPointerAsIndexOp.get(op.source)
            dest_ptr_op = memref.ExtractAlignedPointerAsIndexOp.get(op.destination)

            # Make function call
            func_call = func.Call(
                "snax_dma_1d_transfer",
                [
                    source_ptr_op.aligned_pointer,
                    dest_ptr_op.aligned_pointer,
                    dim_op.result,
                ],
                [],
            )

            # Replace op with function call
            rewriter.insert_op_before_matched_op(
                [zero_const, dim_op, source_ptr_op, dest_ptr_op]
            )
            rewriter.replace_op(op, func_call)

        # Case 2: 2D memrefs with #tsl layouts
        if (
            op.source.type.get_num_dims() == 2
            and op.destination.type.get_num_dims() == 2
            and isinstance(op.source.type.layout, TiledStridedLayoutAttr)
            and isinstance(op.destination.type.layout, TiledStridedLayoutAttr)
        ):
            pass

            ## base addresses ops:
            source_ptr_op = memref.ExtractAlignedPointerAsIndexOp.get(op.source)
            dest_ptr_op = memref.ExtractAlignedPointerAsIndexOp.get(op.destination)

            # assumptions for now:
            # tile sizes are equal between the two memory layouts, only strides differ

            tsl_source = op.source.type.layout.data
            tsl_dest = op.destination.type.layout.data

            # find largest common contiguous block, to be used for dma transfers
            lcb = tsl_source.largest_common_contiguous_block(tsl_dest)

            # all other strides excluded from the contiguous block must be generated
            # using for loops / multi-dimensional dma transfers
            remaining_strides = [
                stride for stride in tsl_source if stride[2] not in lcb
            ]
            # sort the remaining strides by their bound, largest first
            remaining_strides = sorted(
                remaining_strides, key=lambda stride: stride[2].bound, reverse=True
            )
            # map the list to [(src_stride, dest_stride), ...]
            remaining_strides = [
                (stride[2], tsl_dest.get_stride(stride[0], stride[1]))
                for stride in remaining_strides
            ]

            if len(remaining_strides) == 0:
                # is 1d dma transfer
                # TODO
                pass
            elif len(remaining_strides) == 1:
                # is 2d dma transfer
                # TODO
                pass
            elif len(remaining_strides) == 3:
                # use largest loop for the dma transfer, more efficient (probably)
                dma_loop = remaining_strides[0]

                # create dma variables:
                dma_size = arith.Constant.from_int_and_width(
                    lcb[-1].bound, builtin.IndexType()
                )
                dma_stride_src = arith.Constant.from_int_and_width(
                    dma_loop[0].stride, builtin.IndexType()
                )
                dma_stride_dst = arith.Constant.from_int_and_width(
                    dma_loop[1].stride, builtin.IndexType()
                )
                dma_stride_bound = arith.Constant.from_int_and_width(
                    dma_loop[0].bound, builtin.IndexType()
                )

                # remaining strides must be generated using nested loops
                remaining_strides = remaining_strides[1:]

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

                # create list of loop bounds
                lower = arith.Constant.from_int_and_width(0, builtin.IndexType())
                step = arith.Constant.from_int_and_width(1, builtin.IndexType())
                upper = [
                    arith.Constant.from_int_and_width(
                        stride[0].bound, builtin.IndexType()
                    )
                    for stride in remaining_strides
                ]

                # create nested for loop (looping from inner to outer)
                for_loop = scf.For(
                    lower,
                    upper[0],
                    step,
                    [],
                    Region(Block([scf.Yield()], arg_types=(builtin.IndexType(),))),
                )
                for i in range(len(remaining_strides) - 1):
                    for_loop = scf.For(
                        lower,
                        upper[i + 1],
                        step,
                        [],
                        Region(
                            Block(
                                [for_loop, scf.Yield()],
                                arg_types=(builtin.IndexType(),),
                            )
                        ),
                    )

                # insert indexing operations in for loop nest
                for_op = for_loop
                pointer_src = source_ptr_op
                pointer_dst = dest_ptr_op
                for i in range(len(remaining_strides)):
                    next_for_op = for_op.body.block.first_op
                    # insert the ops in the for loop body

                    # source indexing operations:
                    stride_src = arith.Constant.from_int_and_width(
                        remaining_strides[i][0].stride, builtin.IndexType()
                    )
                    increment_src = arith.Muli(
                        for_op.body.block.args[0], stride_src, builtin.IndexType()
                    )
                    pointer_src = arith.Addi(
                        pointer_src, increment_src, builtin.IndexType()
                    )

                    # destination indexing operations:
                    stride_dst = arith.Constant.from_int_and_width(
                        remaining_strides[i][1].stride, builtin.IndexType()
                    )
                    increment_dst = arith.Muli(
                        for_op.body.block.args[0], stride_dst, builtin.IndexType()
                    )
                    pointer_dst = arith.Addi(
                        pointer_dst, increment_dst, builtin.IndexType()
                    )

                    # insert the ops in the for loop body
                    for_op.body.block.insert_ops_before(
                        [
                            stride_src,
                            increment_src,
                            pointer_src,
                            stride_dst,
                            increment_dst,
                            pointer_dst,
                        ],
                        for_op.body.block.first_op,
                    )

                    ## if this is most inner for loop, also insert dma function call
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
                        for_op.body.block.insert_op_before(
                            func_call, for_op.body.block.last_op
                        )

                    # else continue with next for loop
                    else:
                        for_op = next_for_op

                rewriter.insert_op_before_matched_op(
                    [
                        dma_size,
                        dma_stride_bound,
                        dma_stride_src,
                        dma_stride_dst,
                        source_ptr_op,
                        dest_ptr_op,
                        lower,
                        step,
                        *upper,
                    ]
                )
                # rewriter.erase_matched_op()
                rewriter.replace_op(op, for_loop)


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
            PatternRewriteWalker(InsertFunctionCalls()).rewrite_module(op)
            func_decl = func.FuncOp.external(
                "snax_dma_1d_transfer",
                [builtin.IndexType(), builtin.IndexType(), builtin.IndexType()],
                [],
            )
            SymbolTable.insert_or_update(op, func_decl)
            func_decl = func.FuncOp.external(
                "snax_dma_2d_transfer",
                [
                    builtin.IndexType(),
                    builtin.IndexType(),
                    builtin.IndexType(),
                    builtin.IndexType(),
                    builtin.IndexType(),
                    builtin.IndexType(),
                ],
                [],
            )
            SymbolTable.insert_or_update(op, func_decl)
