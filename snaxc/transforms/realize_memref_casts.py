import warnings
from typing import cast

import numpy as np
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, linalg, memref
from xdsl.dialects.memref import MemorySpaceCastOp
from xdsl.ir import Attribute, Operation, OpResult
from xdsl.parser import BytesAttr, DenseIntOrFPElementsAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable
from xdsl.utils.hints import isa

from snaxc.dialects import dart
from snaxc.dialects.snax import LayoutCast
from snaxc.dialects.tsl import TiledStridedLayoutAttr


def is_cast_op(op: Operation) -> bool:
    return isinstance(op, MemorySpaceCastOp) or isinstance(op, LayoutCast)


def transform_constant(
    source: DenseIntOrFPElementsAttr, dest_layout: Attribute
) -> DenseIntOrFPElementsAttr | None:
    """
    Transform a constant op to a new layout.
    """
    if isa(memref_type := source.type, builtin.MemRefType[builtin.FixedBitwidthType]):
        pass

    elif isinstance(source.type, builtin.TensorType):
        memref_type = builtin.MemRefType(
            source.type.get_element_type(), source.type.get_shape()
        )

    else:
        raise NotImplementedError("Can only handle memref and tensor types")

    source_layout = memref_type.layout
    if source_layout == dest_layout:
        # no further transformation needed
        return None

    if not isinstance(source_layout, builtin.NoneAttr):
        warnings.warn("Failed to transform constant op, source layout is not None")
        return None

    if not isinstance(dest_layout, TiledStridedLayoutAttr):
        warnings.warn("failed to transform constant op, dest layout is not tsl")
        return None

    if not dest_layout.data.is_dense():
        warnings.warn("failed to transform constant op, dest layout is not contiguous")
        return None

    if dest_layout.data.is_dynamic():
        warnings.warn("failed to transform constant op, dest layout dynamic")
        return None

    strides = [stride for _, _, stride in dest_layout.data]
    bounds = cast(list[int], [stride.bound for stride in strides])
    order = np.argsort(cast(list[int], [stride.step for stride in strides]))
    # get data:
    values = np.frombuffer(
        source.data.data, dtype=np.dtype(source.get_element_type().format)
    )
    # transform data:
    values = values.reshape(bounds).transpose(order[::-1])
    # update constant op:

    new_type = builtin.MemRefType(
        memref_type.element_type,
        memref_type.shape,
        dest_layout,
        memref_type.memory_space,
    )

    new_value = DenseIntOrFPElementsAttr([new_type, BytesAttr(values.tobytes())])

    return new_value


class RealizeMemrefCasts(RewritePattern):
    """
    A rewrite pattern for realizing memref casts.

    This pattern matches and rewrites MemorySpaceCast and LayoutCast operations
    by performing casting through memref copies and allocations at the right time.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: MemorySpaceCastOp | LayoutCast, rewriter: PatternRewriter
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
        ops_to_add: list[Operation] = []

        # if the source of the memref cast is another layout_cast op,
        # combine them all together
        source_op = op
        while isinstance(source_op.source, OpResult) and isinstance(
            source_op.source.op, MemorySpaceCastOp | LayoutCast
        ):
            source_op = source_op.source.op

        # look for constant op that may be transformed as well
        if (
            isinstance(source_op.source, OpResult)
            and isinstance(const_source := source_op.source.op, arith.ConstantOp)
            and isinstance(op.dest.type, builtin.MemRefType)
        ):
            assert isinstance(const_source.value, DenseIntOrFPElementsAttr)
            new_constant = transform_constant(const_source.value, op.dest.type.layout)
            if new_constant is not None:
                new_constant_op = arith.ConstantOp(new_constant, new_constant.type)
                rewriter.replace_op(const_source, new_constant_op)

        # look for global op that may be transformed as well
        if (
            isinstance(source_op.source, OpResult)
            and isinstance(const_source := source_op.source.op, memref.GetGlobalOp)
            and isinstance(op.dest.type, builtin.MemRefType)
            and isinstance(
                global_op := SymbolTable.lookup_symbol(op, const_source.name_),
                memref.GlobalOp,
            )
            and isa(
                global_op.initial_value,
                DenseIntOrFPElementsAttr[builtin.AnyDenseElement],
            )
        ):
            new_constant = transform_constant(
                global_op.initial_value, op.dest.type.layout
            )
            if new_constant is not None:
                # create new global and get global op with new name
                # global op initial data should be a tensor type:
                new_constant_tensor = DenseIntOrFPElementsAttr(
                    [
                        builtin.TensorType(
                            new_constant.type.get_element_type(),
                            new_constant.type.get_shape(),
                        ),
                        new_constant.data,
                    ]
                )
                new_sym_name = const_source.name_.string_value() + "_transformed"
                new_global_op = memref.GlobalOp.get(
                    builtin.StringAttr(new_sym_name),
                    new_constant.type,
                    new_constant_tensor,
                    global_op.sym_visibility,
                    global_op.constant,
                    global_op.alignment,
                )

                # global get type should inherit only the new layout
                assert isinstance(new_constant.type, builtin.MemRefType)
                get_type = builtin.MemRefType(
                    const_source.memref.type.get_element_type(),
                    const_source.memref.type.get_shape(),
                    new_constant.type.layout,
                    const_source.memref.type.memory_space,
                )
                new_global_get_op = memref.GetGlobalOp(new_sym_name, get_type)

                # insert the global op in the symbol table
                symbol_table_op = global_op.parent_op()
                assert symbol_table_op is not None
                symbol_table = symbol_table_op.get_trait(SymbolTable)
                assert symbol_table is not None

                # insert new global op with new layout
                replaced = symbol_table.insert_or_update(symbol_table_op, new_global_op)
                # assert we have not replaced an existing op
                assert replaced is None

                # delete old global op
                rewriter.erase_op(global_op)

                # replace old global get op with new one
                rewriter.replace_op(const_source, new_global_get_op)

        # now perform casting by inserting memref copies and allocs
        source_type = source_op.source.type
        assert isa(source_type, builtin.MemRefType[Attribute])
        dest_type = op.dest.type
        assert isa(dest_type, builtin.MemRefType[Attribute])

        if source_type == dest_type:
            # canonicalize away unnecessary cast
            op.dest.replace_by(op.source)
            rewriter.erase_matched_op()
            return

        # create allocation

        # create memref.dim operations for dynamic dimensions
        shapes = [x.data for x in dest_type.shape.data]
        dyn_operands: list[Operation] = []
        for i in range(len(shapes)):
            # Dynamic shapes are represented as -1
            if shapes[i] == -1:
                ## create dim op
                index = arith.ConstantOp.from_int_and_width(i, builtin.IndexType())
                dim_op = memref.DimOp.from_source_and_index(
                    source_op.source, index.result
                )
                ops_to_add.extend([index, dim_op])
                dyn_operands.append(dim_op)

        # create alloc op
        alloc_op = memref.AllocOp.get(
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
        assert op.parent
        for use_op in op.parent.walk():
            if use_op not in uses:
                continue
            # check if input
            is_input = False
            if isinstance(use_op, linalg.GenericOp):
                # don't know if input or output, default to yes
                is_input = op.results[0] in use_op.inputs
            elif isinstance(use_op, dart.StreamingRegionOpBase):
                is_input = op.results[0] in use_op.inputs
            else:
                is_input = True
            if is_input:
                # insert copy op
                copy_op = memref.CopyOp(source_op.source, op.dest)
                rewriter.insert_op(copy_op, InsertPoint.before(use_op))
                break

        # insert "copy from" for last use as output
        # walk parent op in reverse order to find last use as output
        for use_op in op.parent.walk(reverse=True):
            if use_op not in uses:
                continue
            # check if input
            is_output = False
            if isinstance(use_op, linalg.GenericOp):
                is_output = op.results[0] in use_op.outputs
            elif isinstance(use_op, dart.StreamingRegionOpBase):
                is_output = op.results[0] in use_op.outputs
            elif isinstance(use_op, func.ReturnOp):
                is_output = False
            else:
                # don't know if input or output, default to yes
                is_output = True
            if is_output:
                # insert copy op
                copy_op = memref.CopyOp(op.dest, source_op.source)
                rewriter.insert_op(copy_op, InsertPoint.after(use_op))
                break

        # insert all ops
        rewriter.replace_matched_op(ops_to_add)


class RealizeMemrefCastsPass(ModulePass):
    name = "realize-memref-casts"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(RealizeMemrefCasts(), walk_reverse=True).rewrite_module(op)
