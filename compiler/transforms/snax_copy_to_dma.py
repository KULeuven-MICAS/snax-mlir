from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, func, scf
from xdsl.dialects.arith import Addi, Constant, Muli
from xdsl.dialects.builtin import (
    FixedBitwidthType,
    IndexType,
    IntAttr,
    IntegerType,
    MemRefType,
    NoneAttr,
    StridedLayoutAttr,
)
from xdsl.dialects.memref import (
    CopyOp,
    Dim,
    ExtractAlignedPointerAsIndexOp,
    ExtractStridedMetaDataOp,
)
from xdsl.ir import Block, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable

from compiler.dialects.tsl import TiledStridedLayoutAttr
from compiler.ir.tsl import TiledStridedLayout


def get_total_size_op(source: SSAValue):
    assert isinstance(source.type, MemRefType)

    # step 1: extract size information to calculate total size
    # this is done by multiplying all the shape dimensions
    ops_to_insert = []
    total_size_op = None
    for dim in range(source.type.get_num_dims()):
        const_op = Constant.from_int_and_width(dim, IndexType())
        ops_to_insert.append(const_op)
        dim_op = Dim.from_source_and_index(source, const_op.result)
        ops_to_insert.append(dim_op)
        if total_size_op is None:
            total_size_op = dim_op
        else:
            total_size_op = Muli(total_size_op.result, dim_op.result, IndexType())
            ops_to_insert.append(total_size_op)

    # step 2: calculate element size to get total size in bytes
    # multiyply the # elements by the (element size // 8) to get the
    # total size in bytes
    element_type: IntegerType = source.type.get_element_type()
    assert isinstance(element_type, FixedBitwidthType)
    element_size_op = Constant.from_int_and_width(element_type.size, IndexType())
    total_size_op = Muli(
        total_size_op.result,
        element_size_op.result,
        IndexType(),
    )
    ops_to_insert.append(element_size_op)
    ops_to_insert.append(total_size_op)

    return ops_to_insert, total_size_op


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

        # get the size of the ops
        ops_to_insert, total_size_op = get_total_size_op(op.source)

        # extract source and destination pointers
        source_ptr_op = ExtractAlignedPointerAsIndexOp.get(op.source)
        dest_ptr_op = ExtractAlignedPointerAsIndexOp.get(op.destination)
        ops_to_insert.append(source_ptr_op)
        ops_to_insert.append(dest_ptr_op)

        # create function call
        func_call = func.Call(
            "snax_dma_1d_transfer",
            [
                source_ptr_op.aligned_pointer,
                dest_ptr_op.aligned_pointer,
                total_size_op.result,
            ],
            [],
        )

        # insert ops and replace op
        rewriter.insert_op_before_matched_op(ops_to_insert)
        rewriter.replace_op(op, func_call)


def extract_strides(memreftype: MemRefType) -> list[int | None]:
    """
    Small helper function to extract the strides from a given memreftype
    with a StridedLayoutAttr or NoneAttr (default row-major) layout.

    Returns:
        List[int] or None: The extracted strides, or None if the strides
        cannot be determined.
    """
    strides: list[int | None]
    if isinstance(memreftype.layout, StridedLayoutAttr):
        strides = [
            x.data if isinstance(x, IntAttr) else None
            for x in memreftype.layout.strides.data
        ]
    elif isinstance(memreftype.layout, NoneAttr):
        # default to row-major layout, construct strides
        # based on shape of the memref type
        strides = [1]
        for size in reversed(memreftype.shape.data[1:]):
            if size.data == -1 or strides[0] is None:
                strides = [None] + strides
            else:
                strides = [size.data * strides[0]] + strides
    else:
        raise NotImplementedError("This memref layout type is not handled yet.")
    return strides


def extract_offset(memreftype: MemRefType):
    """
    Small helper function to extract the offset from a given memreftype
    with a StridedLayoutAttr or NoneAttr (default row-major) layout.

    Returns:
        int | None: The extracted offset or None if it is dynamic.
    """
    if isinstance(memreftype.layout, StridedLayoutAttr):
        # Dynamic offset
        if isinstance(memreftype.layout.offset, NoneAttr):
            return None
        return memreftype.layout.offset.data

    return 0


class TransformDMA(RewritePattern):
    """Look for memref copy operations with TSL layout and insert snitch DMA calls"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CopyOp, rewriter: PatternRewriter):
        # both operands should be memref types
        if not (
            isinstance(op.source.type, MemRefType)
            and isinstance(op.destination.type, MemRefType)
        ):
            return

        # both operands should be of equal shape and integer element type
        if any(
            [
                not op.destination.type.get_shape() == op.source.type.get_shape(),
                not op.destination.type.get_element_type()
                == op.source.type.get_element_type(),
                not isinstance(op.source.type.get_element_type(), IntegerType),
            ]
        ):
            return

        # if source is not tsl, construct representation:
        if isinstance(op.source.type.layout, TiledStridedLayoutAttr):
            tsl_source = op.source.type.layout
        else:
            strides = extract_strides(op.source.type)
            tile_bounds: list[list[int | None]]
            offset = extract_offset(op.source.type)
            if not strides:
                return
            if isinstance(op.destination.type.layout, TiledStridedLayoutAttr):
                # if destination is tsl, use those tile sizes
                tile_bounds = op.destination.type.layout.data.tile_bounds()
            else:
                # otherwise, shape can be used as single-dimension tile sizes
                # change dynamic size -1 to None
                tile_bounds = [
                    [x.data] if x.data > 0 else [None]
                    for x in op.source.type.shape.data
                ]
            tsl_source = TiledStridedLayoutAttr(
                TiledStridedLayout.from_strides(strides, tile_bounds, offset)
            )

        # if dest is not tsl, construct representation:
        if isinstance(op.destination.type.layout, TiledStridedLayoutAttr):
            tsl_dest = op.destination.type.layout
        else:
            strides = extract_strides(op.destination.type)
            tile_bounds: list[list[int | None]]
            offset = extract_offset(op.destination.type)
            if not strides:
                return
            if isinstance(op.source.type.layout, TiledStridedLayoutAttr):
                # if destination is tsl, use those tile sizes
                tile_bounds = op.source.type.layout.data.tile_bounds()
            else:
                # otherwise, shape can be used as single-dimension tile sizes
                # change dynamic size -1 to None
                tile_bounds = [
                    [x.data] if x.data > 0 else [None]
                    for x in op.source.type.shape.data
                ]
            tsl_dest = TiledStridedLayoutAttr(
                TiledStridedLayout.from_strides(strides, tile_bounds, offset)
            )

        # list of all ops that need to be inserted
        ops_to_insert = []

        # step 1: extract base addresses
        pointer_src = ExtractAlignedPointerAsIndexOp.get(op.source)
        pointer_dst = ExtractAlignedPointerAsIndexOp.get(op.destination)
        ops_to_insert.append(pointer_src)
        ops_to_insert.append(pointer_dst)

        # apply offset if it is not zero
        if tsl_source.data.offset != 0:
            # Calculate number of bytes in type
            assert isinstance(op.source.type.element_type, FixedBitwidthType)
            el_bytes = op.source.type.element_type.size
            el_bytes_op = Constant.from_int_and_width(el_bytes, IndexType())
            # Dynamic offset
            if tsl_source.data.offset is None:
                # dynamic offsets for tsl is TODO
                assert isinstance(op.source.type.layout, StridedLayoutAttr)
                offset_op = ExtractStridedMetaDataOp(op.source)
                offset = offset_op.offset
            else:
                offset_op = Constant.from_int_and_width(
                    tsl_source.data.offset, IndexType()
                )
                offset = offset_op.result

            calc_offset_op = Muli(el_bytes_op, offset, IndexType())
            pointer_src = Addi(pointer_src, calc_offset_op, IndexType())
            ops_to_insert.extend([offset_op, el_bytes_op, calc_offset_op, pointer_src])

        if tsl_dest.data.offset != 0:
            # Dynamic offset
            assert isinstance(op.destination.type.element_type, FixedBitwidthType)
            el_bytes = op.destination.type.element_type.size
            el_bytes_op = Constant.from_int_and_width(el_bytes, IndexType())
            if tsl_dest.data.offset is None:
                assert isinstance(op.destination.type.layout, StridedLayoutAttr)
                offset_op = ExtractStridedMetaDataOp(op.destination)
                offset = offset_op.offset
            else:
                # Multiplication with el_bytes already happens statically with extract_offset()
                offset_op = Constant.from_int_and_width(
                    tsl_dest.data.offset, IndexType()
                )
                offset = offset_op.result
            calc_offset_op = Muli(el_bytes_op, offset, IndexType())
            pointer_dst = Addi(pointer_dst, calc_offset_op, IndexType())
            ops_to_insert.extend([offset_op, el_bytes_op, calc_offset_op, pointer_dst])

        # step 2: find largest common contiguous block, to be used for dma transfers
        assert isinstance(op.source.type.element_type, FixedBitwidthType)
        lcb = tsl_source.data.largest_common_contiguous_block(tsl_dest.data)

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
        ops_to_add, step_ops_src = tsl_source.get_step_ops(
            bound_ops, op.source, in_bytes=True
        )
        ops_to_insert.extend(ops_to_add)

        # generate the step ops for the destination tsl
        ops_to_add, step_ops_dst = tsl_dest.get_step_ops(
            bound_ops, op.destination, in_bytes=True
        )
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
            # is actually 1d dma transfer, calculate size of transfer:
            ops_to_add, total_size_op = get_total_size_op(op.source)
            ops_to_insert.extend(ops_to_add)

            # create function call
            func_call = func.Call(
                "snax_dma_1d_transfer", [pointer_src, pointer_dst, total_size_op], []
            )

            # insert ops and replace op
            rewriter.insert_op_before_matched_op(ops_to_insert)
            rewriter.replace_op(op, func_call)
            return
        else:
            dma_loop = remaining_strides.pop(0)

        # if my reasoning is correct, if there are remaining strides,
        # then the lcb cannot be dynamic
        assert lcb[-1].bound is not None
        assert lcb[-1].step is not None

        assert isinstance(op.source.type.element_type, FixedBitwidthType)
        el_bytes = op.source.type.element_type.size
        dma_size = Constant.from_int_and_width(
            lcb[-1].bound * lcb[-1].step * el_bytes, IndexType()
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
                    pointer_src,
                    pointer_dst,
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
        PatternRewriteWalker(TransformDMA()).rewrite_module(op)

        if any(
            isinstance(op_in_module, func.Call)
            and op_in_module.callee.root_reference.data == "snax_dma_1d_transfer"
            for op_in_module in op.walk()
        ):
            func_decl = func.FuncOp.external(
                "snax_dma_1d_transfer", 3 * [builtin.IndexType()], []
            )
            SymbolTable.insert_or_update(op, func_decl)

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
