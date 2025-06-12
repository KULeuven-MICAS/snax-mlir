from xdsl.context import Context
from xdsl.dialects import builtin, memref
from xdsl.dialects.arith import AddiOp, ConstantOp, MuliOp, SubiOp
from xdsl.dialects.builtin import (
    FixedBitwidthType,
    IndexType,
    NoneAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Operation, OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from snaxc.dialects import snax
from snaxc.dialects.tsl import TiledStridedLayoutAttr
from snaxc.util.snax_memory import L1


class AllocOpRewrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, alloc_op: memref.AllocOp, rewriter: PatternRewriter):
        """Swap memref.alloc op with snax.alloc, for now, we support
        NoneType layouts and TSL Layouts, and a memory space of L1"""

        # get the element type
        element_type = alloc_op.memref.type.get_element_type()

        if not isinstance(element_type, builtin.IntegerType | builtin.AnyFloat):
            return

        # get the memory space
        memory_space = alloc_op.memref.type.memory_space

        # if the memory space is not L1, conversion to snax is not possible
        if memory_space != L1.attribute:
            return

        # get the layout
        layout = alloc_op.memref.type.layout

        # create an operation to get the # bytes that needs
        # to be allocated
        total_size_op = None
        ops_to_add: list[Operation] = []

        # generate the list of shape ops
        # either these are constant and must be created,
        # or they are already present in the memref.alloc
        # operation arguments
        shape_ops: list[Operation] = []
        alloc_args: list[Operation] = []
        for size in alloc_op.dynamic_sizes:
            assert isinstance(size, OpResult)
            alloc_args.append(size.op)

        for shape in alloc_op.memref.type.shape.data:
            if shape.data == -1:
                # dynamic op
                shape_ops.append(alloc_args.pop(0))
            else:
                # constant op
                shape_op = ConstantOp.from_int_and_width(shape.data, IndexType())
                ops_to_add.append(shape_op)
                shape_ops.append(shape_op)

        shape_ops_arg: list[Operation] = [x for x in shape_ops]

        if isinstance(layout, NoneAttr):
            # get size based on shape
            shape = alloc_op.memref.type.shape

            # multiply all the dimensions with the element width
            # to get the size we need to allocate
            assert isinstance(element_type, FixedBitwidthType)
            element_size_op = ConstantOp.from_int_and_width(element_type.size, IndexType())
            total_size_op = element_size_op
            ops_to_add.append(element_size_op)

            for dim in range(len(shape)):
                # we can assume all shapes are static for now
                shape_op = shape_ops.pop(0)
                total_size_op = MuliOp(shape_op, total_size_op)
                ops_to_add.append(total_size_op)

        if isinstance(layout, TiledStridedLayoutAttr):
            # to get the entire size needed for a TSL layout,
            # we need to compute the following for all strides:
            # sum_i( (bound_i - 1) * step_i) + 1

            # use shape ops to generate tsl bound ops
            insert_ops, bound_ops = layout.get_bound_ops(shape_ops)
            ops_to_add.extend(insert_ops)
            insert_ops, step_ops = layout.get_step_ops(bound_ops, alloc_op.memref, in_bytes=True)
            ops_to_add.extend(insert_ops)

            cst_1 = ConstantOp.from_int_and_width(1, IndexType())
            ops_to_add.append(cst_1)
            total_size_op = cst_1

            stride_max = ConstantOp.from_int_and_width(0, IndexType())
            ops_to_add.append(stride_max)

            # iterate over keys, values of bound_ops:
            # to calculate sum_i( (bound_i - 1) * step_i)
            for (dim, depth), bound_op in bound_ops.items():
                bound_op_minus_1 = SubiOp(bound_op, cst_1)
                stride_op = step_ops[(dim, depth)]
                mul_op = MuliOp(bound_op_minus_1, stride_op)
                stride_max = AddiOp(stride_max, mul_op)
                ops_to_add.extend([bound_op_minus_1, mul_op, stride_max])

            # add final + element_width
            assert isinstance(element_type, FixedBitwidthType)
            element_size_op = ConstantOp.from_int_and_width(element_type.size, IndexType())
            stride_max = AddiOp(stride_max, element_size_op)
            ops_to_add.extend([element_size_op, stride_max])

            total_size_op = MuliOp(total_size_op, stride_max)
            ops_to_add.append(total_size_op)

            # add offset
            assert layout.data.offset is not None
            offset_op = ConstantOp.from_int_and_width(layout.data.offset, IndexType())
            offset_bytes_op = MuliOp(offset_op, element_size_op)
            total_size_op = AddiOp(total_size_op, offset_bytes_op)
            ops_to_add.extend([offset_op, offset_bytes_op, total_size_op])

        if total_size_op is None:
            return

        # create snax alloc op
        snax_alloc = snax.Alloc(
            alloc_op.memref.type.get_num_dims(),
            total_size_op,
            shape_ops_arg,
            memory_space,
            alloc_op.alignment,
        )
        conversion_cast_op = UnrealizedConversionCastOp.get([snax_alloc], [alloc_op.memref.type])
        rewriter.replace_matched_op(
            [*ops_to_add, snax_alloc, conversion_cast_op],
            new_results=conversion_cast_op.outputs,
        )


class MemrefToSNAX(ModulePass):
    name = "memref-to-snax"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(AllocOpRewrite()).rewrite_module(op)
