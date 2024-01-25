from xdsl.dialects import arith, builtin, func, scf
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


class MatchSimpleCopy(RewritePattern):
    """
    Looks for simple dense memref copy (without layout information)
    operations and inserts a snitch 1d dma call. The size of the memrefs
    is determinded by the shape and element type of the memref.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CopyOp, rewriter: PatternRewriter):
        # only works on memrefs with nonetype layouts and equal shape and element type
        if any(
            [
                not isinstance(op.source.type, MemRefType),
                not isinstance(op.destination.type, MemRefType),
                not isinstance(op.source.type.layout, NoneAttr),
                not isinstance(op.destination.type.layout, NoneAttr),
                not op.destination.type.get_shape() == op.source.type.get_shape(),
                not op.destination.type.get_element_type()
                == op.source.type.get_element_type(),
                not isinstance(op.source.type.get_element_type(), IntegerType),
            ]
        ):
            return

        # step 1: extract size information to calculate total size
        # this is done by multiplying all the shape dimensions
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
                ops_to_insert.append(total_size_op)

        # step 2: calculate element size to get total size in bytes
        # multiyply the # elements by the (element size // 8) to get the
        # total size in bytes
        element_type: IntegerType = op.source.type.get_element_type()
        assert element_type.width.data % 8 == 0
        element_size = element_type.width.data // 8
        element_size_op = Constant.from_int_and_width(element_size, IndexType())
        total_size_op = Muli(
            total_size_op.result,
            element_size_op.result,
            IndexType(),
        )
        ops_to_insert.append(element_size_op)
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
        if any(
            [
                not isinstance(op.source.type, MemRefType),
                not isinstance(op.destination.type, MemRefType),
                not isinstance(op.source.type.layout, TiledStridedLayoutAttr),
                not isinstance(op.destination.type.layout, TiledStridedLayoutAttr),
                not op.destination.type.get_shape() == op.source.type.get_shape(),
                not op.destination.type.get_element_type()
                == op.source.type.get_element_type(),
                not isinstance(op.source.type.get_element_type(), IntegerType),
            ]
        ):
            return

        # list of all ops that need to be inserted
        ops_to_insert = []

        # step 1: extract base addresses
        pointer_src = ExtractAlignedPointerAsIndexOp.get(op.source)
        pointer_dst = ExtractAlignedPointerAsIndexOp.get(op.destination)
        ops_to_insert.append(pointer_src)
        ops_to_insert.append(pointer_dst)

        # step 2: find largest common contiguous block, to be used for dma transfers
        tsl_source = op.source.type.layout
        tsl_dest = op.destination.type.layout

        # lcb is completely static
        assert op.source.element_type.width.data % 8 == 0
        lcb = tsl_source.data.largest_common_contiguous_block(
            tsl_dest.data, op.source.type.element_type.width.data // 8
        )

        # step 3: generate ops for the strides and bounds of the TSL
        # except for the strides in the LCB, the other strides are used
        # to calculate the bounds of the nested for loops for these Strides
        # we need to generate the ops for the bounds and steps of these Strides.
        # if these values are known beforehand, we can insert them as constants
        # if not, we need to generate the ops for the calculation of these values

        # to do this, we will generate a dict with the following structure:
        #     (dim, depth) : {
        #         "stride_src": Stride
        #         "stride_dst": Stride
        #         "bound_op": None | Op
        #         "step_src_op": None | Op
        #         "step_dst_op": None | Op
        #     }
        # In this dict, dim and depth refer to the dimension and depth of the
        # current Stride. stride_src and stride_dst contain  the Stride objects
        # for the source and destination. bound_op, step_src_op and step_dst_op
        # contain the ops for the bound, step_src and step_dst of the Stride,
        # which we need to generate.

        remaining_strides = {}

        # first, generate the bound ops, we only need to do this for the source
        # tsl, since the destination tsl has the same bounds (constraint)
        ops_to_add, bound_ops = tsl_source.get_bound_ops(op.source)
        ops_to_insert.extend(ops_to_add)

        # generate the step ops for the source tsl
        ops_to_add, step_ops_src = tsl_source.get_step_ops(bound_ops)
        ops_to_insert.extend(ops_to_add)

        # generate the step ops for the destination tsl
        ops_to_add, step_ops_dst = tsl_dest.get_step_ops(bound_ops)
        ops_to_insert.extend(ops_to_add)

        # construct the dict. we only need the strides not yet present in the lcb
        for key in bound_ops.keys():
            stride = tsl_source.data.get_stride(*key)
            if stride not in lcb:
                remaining_strides[key] = {
                    "stride_src": tsl_source.data.get_stride(*key),
                    "stride_dst": tsl_dest.data.get_stride(*key),
                    "bound_op": bound_ops[key],
                    "step_src_op": step_ops_src[key],
                    "step_dst_op": step_ops_dst[key],
                }

        # sort the remaining strides
        # we want the nested for loop to be sorted by descending bound
        remaining_strides = sorted(
            remaining_strides.values(),
            key=lambda x: x["stride_src"].bound if x["stride_src"].bound else 0,
            reverse=True,
        )

        # step 4: generate variables for 2D dma transfer
        if len(remaining_strides) == 0:
            # is actually 1d dma transfer, fake 2d transfer for experiments
            one_op = arith.Constant.from_int_and_width(1, IndexType())
            ops_to_insert.append(one_op)
            dma_loop = {
                "step_src_op": one_op,
                "step_dst_op": one_op,
                "bound_op": one_op,
            }
        else:
            dma_loop = remaining_strides.pop(0)

        dma_size = Constant.from_int_and_width(
            lcb[-1].bound * lcb[-1].step, IndexType()
        )
        dma_stride_src = dma_loop["step_src_op"]
        dma_stride_dst = dma_loop["step_dst_op"]
        dma_stride_bound = dma_loop["bound_op"]
        ops_to_insert.extend([dma_size])

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
        upper = [stride["bound_op"] for stride in remaining_strides]

        ops_to_insert.extend([lower, step])

        # step 6.2: create nested for loop (looping from inner to outer)
        # innermost for loop has empty region
        empty_region = Region(Block([scf.Yield()], arg_types=(IndexType(),)))
        for_loop = scf.For(lower, upper[-1], step, [], empty_region)

        for i in range(len(remaining_strides) - 1):
            # other for loops have a region with the previous for loop as body
            region = Region(Block([for_loop, scf.Yield()], arg_types=(IndexType(),)))
            for_loop = scf.For(
                lower, upper[len(remaining_strides) - 2 - i], step, [], region
            )

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
            stride_src = remaining_strides[i]["step_src_op"]
            increment_src = Muli(for_loop.body.block.args[0], stride_src, IndexType())
            pointer_src = Addi(pointer_src, increment_src, IndexType())
            ops_to_insert_for_loop.extend([increment_src, pointer_src])

            # destination indexing operations:
            stride_dst = remaining_strides[i]["step_dst_op"]
            increment_dst = Muli(for_loop.body.block.args[0], stride_dst, IndexType())
            pointer_dst = Addi(pointer_dst, increment_dst, IndexType())
            ops_to_insert_for_loop.extend([increment_dst, pointer_dst])

            # insert the ops in the for loop body
            for_loop.body.block.insert_ops_before(
                ops_to_insert_for_loop, for_loop.body.block.first_op
            )

            # if this is innermost for loop, also insert dma function call
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
        PatternRewriteWalker(MatchSimpleCopy()).rewrite_module(op)
        if any(
            isinstance(op_in_module, func.Call)
            and op_in_module.callee.root_reference.data == "snax_dma_1d_transfer"
            for op_in_module in op.walk()
        ):
            func_decl = func.FuncOp.external(
                "snax_dma_1d_transfer", 3 * [builtin.IndexType()], []
            )
            SymbolTable.insert_or_update(op, func_decl)

        PatternRewriteWalker(TransformDMA()).rewrite_module(op)
        if any(
            isinstance(op_in_module, func.Call)
            and op_in_module.callee.root_reference.data == "snax_dma_2d_transfer"
            for op_in_module in op.walk()
        ):
            func_decl = func.FuncOp.external(
                "snax_dma_2d_transfer", 6 * [builtin.IndexType()], []
            )
            SymbolTable.insert_or_update(op, func_decl)

        # remove dead code because a lot of unused stuff is generated
        # e.g. the ops for the bounds and steps of the Strides which
        # may not be used at the end
        # dce(op)
        # TODO because there are some things missing in xdsl
