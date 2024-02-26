from xdsl.dialects import builtin, memref
from xdsl.dialects.arith import Addi, Constant, Muli, Subi
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
        """Swap memref.alloc op with snax.alloc, for now, we support
        NoneType layouts and TSL Layouts, and a memory space of 1 (=L1)"""

        # get the memref type
        memref_type: memref.MemRefType = alloc_op.memref.type

        # get the element type
        element_type = memref_type.get_element_type()

        if not isinstance(element_type, builtin.IntegerType | builtin.AnyFloat):
            return

        # get the memory space
        memory_space = memref_type.memory_space

        # if the memory space is not 1, conversion to snax is not possible
        if isinstance(memory_space, NoneAttr) or memory_space.value.data != 1:
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

        shape_ops_arg = [x for x in shape_ops]

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
            # we need to compute the following for all strides:
            # sum_i( (bound_i - 1) * step_i) + 1

            # use shape ops to generate tsl bound ops
            insert_ops, bound_ops = layout.get_bound_ops(shape_ops)
            ops_to_add.extend(insert_ops)
            insert_ops, step_ops = layout.get_step_ops(bound_ops)
            ops_to_add.extend(insert_ops)

            # for tsl, element_size = 1 byte by definition,
            # element width is encoded in the strides of the tsl
            cst_1 = Constant.from_int_and_width(1, IndexType())
            ops_to_add.append(cst_1)
            total_size_op = cst_1

            stride_max = Constant.from_int_and_width(0, IndexType())
            ops_to_add.append(stride_max)

            # iterate over keys, values of bound_ops:
            # to calculate sum_i( (bound_i - 1) * step_i)
            for (dim, depth), bound_op in bound_ops.items():
                bound_op_min_1 = Subi(bound_op, cst_1)
                stride_op = step_ops[(dim, depth)]
                mul_op = Muli(bound_op_min_1, stride_op)
                stride_max = Addi(stride_max, mul_op)
                ops_to_add.extend([bound_op_min_1, mul_op, stride_max])

            # add final + element_width
            if isinstance(element_type, builtin.AnyFloat):
                element_width = element_type.get_bitwidth
            else:
                element_width = element_type.width.data
            assert element_width % 8 == 0
            element_size = element_width // 8
            element_size_op = Constant.from_int_and_width(element_size, IndexType())
            stride_max = Addi(stride_max, element_size_op)
            ops_to_add.extend([element_size_op, stride_max])

            total_size_op = Muli(total_size_op, stride_max)
            ops_to_add.append(total_size_op)

        if total_size_op is None:
            return

        # create snax alloc op
        snax_alloc = snax.Alloc(
            memref_type.get_num_dims(),
            total_size_op,
            shape_ops_arg,
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
