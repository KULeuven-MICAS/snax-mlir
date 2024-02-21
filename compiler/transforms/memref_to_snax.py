from xdsl.dialects import builtin, memref
from xdsl.dialects.arith import Addi, Constant, Muli
from xdsl.dialects.builtin import IndexType, NoneAttr, UnrealizedConversionCastOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.dialects import snax
from compiler.dialects.tsl import TiledStridedLayoutAttr


class AllocOpRewrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, alloc_op: memref.Alloc, rewriter: PatternRewriter):
        """Swap memref.alloc op with snax.alloc, for now, we suppport
        NoneType layouts and TSL Layouts"""

        # get the memref type
        memref_type: memref.MemRefType = alloc_op.memref.type

        # get the element type
        element_type = memref_type.get_element_type()

        if not isinstance(element_type, builtin.IntegerType | builtin.AnyFloat):
            return

        # get the memory space
        memory_space = memref_type.memory_space

        # if the memory space is not defined, conversion to snax is not possible
        if isinstance(memory_space, NoneAttr):
            return

        # get the layout
        layout = memref_type.layout

        # create an operation to get the # bytes that needs
        # to be allocated
        total_size_op = None
        ops_to_add = []

        # generate the list of shape ops
        # either these are constant and must be created,
        # or they are already present in the memref.alloc
        # operation arguments
        shape_ops = []
        alloc_args = [x.op for x in alloc_op.dynamic_sizes]

        for shape in memref_type.shape.data:
            if shape.data == -1:
                # dynamic op
                shape_ops.append(alloc_args.pop(0))
            else:
                # constant op
                shape_op = Constant.from_int_and_width(shape.data, IndexType())
                ops_to_add.append(shape_op)
                shape_ops.append(shape_op)

        if isinstance(layout, NoneAttr):
            # get size based on shape
            shape = memref_type.shape

            # multiply all the dimensions with the element width
            # to get the size we need to allocate
            assert element_type.width.data % 8 == 0
            element_size = element_type.width.data // 8
            element_size_op = Constant.from_int_and_width(element_size, IndexType())
            total_size_op = element_size_op
            ops_to_add.append(element_size_op)

            for dim in range(len(shape)):
                # we can assume all shapes are static for now
                shape_op = shape_ops.pop(0)
                total_size_op = Muli(shape_op, total_size_op)
                ops_to_add.append(total_size_op)

        if isinstance(layout, TiledStridedLayoutAttr):
            # to get the entire size needed for a TSL layout,
            # we need to take the sum of all bounds multiplied with
            # their respective strides

            # use shape ops to generate tsl bound ops
            insert_ops, bound_ops = layout.get_bound_ops(shape_ops)
            ops_to_add.extend(insert_ops)
            insert_ops, step_ops = layout.get_step_ops(bound_ops)
            ops_to_add.extend(insert_ops)

            # start with the element size width
            element_size = element_type.width.data // 8
            element_size_op = Constant.from_int_and_width(element_size, IndexType())
            total_size_op = element_size_op
            ops_to_add.append(element_size_op)

            stride_max = Constant.from_int_and_width(0, IndexType())
            ops_to_add.append(stride_max)

            # iterate over keys, values of bound_ops:
            for (dim, depth), bound_op in bound_ops.items():
                stride_op = step_ops[(dim, depth)]
                mul_op = Muli(bound_op, stride_op)
                stride_max = Addi(stride_max, mul_op)
                ops_to_add.extend([mul_op, stride_max])

            total_size_op = Muli(total_size_op, stride_max)
            ops_to_add.append(total_size_op)

        if total_size_op is None:
            return

        # create snax alloc op
        snax_alloc = snax.Alloc(
            memref_type.get_num_dims(),
            total_size_op,
            memory_space,
            alloc_op.alignment,
            element_type,
        )
        conversion_cast_op = UnrealizedConversionCastOp.get([snax_alloc], memref_type)
        rewriter.replace_matched_op(
            [*ops_to_add, snax_alloc, conversion_cast_op],
            new_results=conversion_cast_op.outputs,
        )


class MemrefToSNAX(ModulePass):
    name = "memref-to-snax"

    def apply(self, ctx: MLContext, module: builtin.ModuleOp) -> None:
        PatternRewriteWalker(AllocOpRewrite()).rewrite_module(module)
