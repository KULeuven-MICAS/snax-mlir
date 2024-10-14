from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, func, linalg, memref
from xdsl.dialects.memref import MemorySpaceCast
from xdsl.ir import Operation, OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.dialects import stream
from compiler.dialects.snax import LayoutCast


def is_cast_op(op: Operation) -> bool:
    return isinstance(op, MemorySpaceCast) or isinstance(op, LayoutCast)


class RealizeMemrefCasts(RewritePattern):
    """
    A rewrite pattern for realizing memref casts.

    This pattern matches and rewrites MemorySpaceCast and LayoutCast operations
    by performing casting through memref copies and allocations at the right time.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: MemorySpaceCast | LayoutCast, rewriter: PatternRewriter
    ):
        # if the casting is not used anymore (perhaps made useless by previous
        # cast realizations), we do not need to do anything. dce will remove it later
        if not op.dest.uses:
            return

        # due to previous passes, it is common for multiple memref casting
        # ops to be chained together. For now all the transformations are handled
        # by the DMA which can access all memory spaces, and handle all transformations
        # so we can fuse all the casting operations together.

        # keep track of ops to add
        ops_to_add = []

        # if the source of the memref cast is another layout_cast op,
        # combine them all together
        source_op = op
        while isinstance(source_op.source, OpResult) and isinstance(
            source_op.source.op, MemorySpaceCast | LayoutCast
        ):
            source_op = source_op.source.op

        # now perform casting by inserting memref copies and allocs
        source_type = source_op.source.type
        assert isinstance(source_type, memref.MemRefType)
        dest_type = op.dest.type
        assert isinstance(dest_type, memref.MemRefType)

        # create allocation

        # create memref.dim operations for dynamic dimensions
        shapes = [x.data for x in dest_type.shape.data]
        dyn_operands = []
        for i in range(len(shapes)):
            # Dynamic shapes are represented as -1
            if shapes[i] == -1:
                ## create dim op
                index = arith.Constant.from_int_and_width(i, builtin.IndexType())
                dim_op = memref.Dim.from_source_and_index(
                    source_op.source, index.result
                )
                ops_to_add.extend([index, dim_op])
                dyn_operands.append(dim_op)

        # create alloc op
        alloc_op = memref.Alloc.get(
            dest_type.get_element_type(),
            64,  # default 64 alignment
            dest_type.get_shape(),
            dynamic_sizes=dyn_operands,
            layout=dest_type.layout,
            memory_space=dest_type.memory_space,
        )
        ops_to_add.append(alloc_op)

        # Insert copy ops if newly allocated memref is used as
        # input or output, list to visit all uses of allocated memrefs:
        uses = [x.operation for x in op.dest.uses]

        # insert "copy to" for first use as input
        # walk parent op in order to find first use as input
        for use_op in op.parent.walk():
            if use_op not in uses:
                continue
            # check if input
            is_input = False
            if isinstance(use_op, linalg.Generic):
                # don't know if input or output, default to yes
                is_input = op.results[0] in use_op.inputs
            elif isinstance(use_op, stream.StreamingRegionOp):
                is_input = op.results[0] in use_op.inputs
            else:
                is_input = True
            if is_input:
                # insert copy op
                copy_op = memref.CopyOp(source_op.source, op.dest)
                rewriter.insert_op_before(copy_op, use_op)
                break

        # insert "copy from" for last use as output
        # walk parent op in reverse order to find last use as output
        for use_op in op.parent.walk(reverse=True):
            if use_op not in uses:
                continue
            # check if input
            is_output = False
            if isinstance(use_op, linalg.Generic):
                is_output = op.results[0] in use_op.outputs
            elif isinstance(use_op, stream.StreamingRegionOp):
                is_output = op.results[0] in use_op.outputs
            elif isinstance(use_op, func.Return):
                is_output = False
            else:
                # don't know if input or output, default to yes
                is_output = True
            if is_output:
                # insert copy op
                copy_op = memref.CopyOp(op.dest, source_op.source)
                rewriter.insert_op_after(copy_op, use_op)
                break

        # insert all ops
        rewriter.replace_matched_op(ops_to_add)


class RealizeMemrefCastsPass(ModulePass):
    name = "realize-memref-casts"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(RealizeMemrefCasts(), walk_reverse=True).rewrite_module(op)
