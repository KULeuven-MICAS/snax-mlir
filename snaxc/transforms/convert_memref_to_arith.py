from math import prod
from typing import cast

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.arith import AddiOp, ConstantOp, DivUIOp, MuliOp
from xdsl.dialects.builtin import FixedBitwidthType, IndexType
from xdsl.dialects.memref import ExtractAlignedPointerAsIndexOp, SubviewOp
from xdsl.ir import Attribute, Operation, OpResult
from xdsl.parser import MemRefType
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa

from snaxc.dialects.tsl import TiledStridedLayoutAttr


class LowerExtractAlignedPointerOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExtractAlignedPointerAsIndexOp, rewriter: PatternRewriter):
        if not isa(op.source.type, MemRefType[Attribute]):
            return
        if not isinstance(op.source, OpResult):
            return
        if not isinstance(subview := op.source.op, SubviewOp):
            return
        assert isa(source_type := subview.source.type, MemRefType[Attribute])
        if not isinstance(source_type.layout, TiledStridedLayoutAttr):
            return
        # TODO: remove cast once xdsl typing issue is resolved
        layout = cast(TiledStridedLayoutAttr, source_type.layout)
        dynamic_index_list = [
            i for i, offset in enumerate(subview.static_offsets.get_values()) if offset == SubviewOp.DYNAMIC_INDEX
        ]
        ops_to_add: list[Operation] = []
        aligned_pointer = ExtractAlignedPointerAsIndexOp.get(subview.source)
        ops_to_add.append(aligned_pointer)
        element_type = source_type.get_element_type()
        assert isinstance(element_type, FixedBitwidthType)
        bytes_op = ConstantOp.from_int_and_width(element_type.size, IndexType())
        ops_to_add.append(bytes_op)
        for offset, index in zip(subview.offsets, dynamic_index_list):
            stride = layout.data.tstrides[index].strides[0].step
            assert stride is not None
            stride_op = ConstantOp.from_int_and_width(stride, IndexType())
            stride_bytes_op = MuliOp(stride_op, bytes_op)
            bound = prod(cast(int, stride.bound) for stride in layout.data.tstrides[index].strides[1:])
            assert bound is not None
            bound_op = ConstantOp.from_int_and_width(bound, IndexType())
            offset_div = DivUIOp(offset, bound_op)
            offset_op = MuliOp(offset_div, stride_bytes_op)
            aligned_pointer = AddiOp(aligned_pointer, offset_op)
            ops_to_add.extend([stride_op, stride_bytes_op, bound_op, offset_div, offset_op, aligned_pointer])
        rewriter.replace_matched_op(ops_to_add)


class ConvertMemrefToArithPass(ModulePass):
    name = "convert-memref-to-arith"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(LowerExtractAlignedPointerOp()).rewrite_module(op)
