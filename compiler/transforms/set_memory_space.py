from xdsl.dialects import arith, builtin, func, linalg, memref
from xdsl.ir import MLContext, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class InitFuncMemorySpace(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        """Add a default (0 : i32) memory space to memrefs used in the function
        that do not have a memory space specified yet"""

        # Function must be public
        if op.sym_visibility.data != "public":
            return

        # Function must have memref arguments with an undefined memory space
        if not any(
            [
                isinstance(x, memref.MemRefType)
                and isinstance(x.memory_space, builtin.NoneAttr)
                for x in [*op.function_type.inputs, *op.function_type.outputs]
            ]
        ):
            return

        # Mapping function to assign default memory space (0 : i32)
        def change_to_memory_space(t):
            if isinstance(t, memref.MemRefType):
                if isinstance(t.memory_space, builtin.NoneAttr):
                    return memref.MemRefType.from_element_type_and_shape(
                        t.element_type,
                        t.get_shape(),
                        t.layout,
                        builtin.IntegerAttr(0, builtin.i32),
                    )
            return t

        # Define new function type with updated inputs and outputs
        # mapped to a default memory space
        new_function_type = builtin.FunctionType.from_lists(
            map(change_to_memory_space, op.function_type.inputs),
            (op.function_type.outputs),
        )

        # Change region of function to use new argument types
        for arg in op.args:
            arg.type = change_to_memory_space(arg.type)

        # Define new function op with new type and copy region contents
        new_op = func.FuncOp(
            op.sym_name.data,
            new_function_type,
            region=rewriter.move_region_contents_to_new_regions(op.regions[0]),
            visibility=op.sym_visibility,
        )

        # Replace function op
        rewriter.replace_matched_op(new_op)


class InitLinalgMemorySpace(RewritePattern):
    """Walk through dispatchable operations (just linalg.Generic for now)
    and change them to use only memrefs in memory space (1 : i32). If they
    currently use a memref in a different memory adress space, insert a
    memref.memory_space_cast operation to convert the two"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        # Op must have memref arguments with memory space not equal to 1
        if not any(
            [
                isinstance(x.type, memref.MemRefType)
                and isinstance(x.type.memory_space, builtin.IntegerAttr)
                and x.type.memory_space.value.data != 1
                for x in op.inputs
            ]
        ):
            return

        # Function to find/create casting operand if it is necessary
        def get_cast_op(operand) -> None | memref.MemorySpaceCast:
            # check if cast is required: must be a memref in wrong memory space
            if not isinstance(operand, SSAValue):
                return None
            if not isinstance(operand.type, memref.MemRefType):
                return None
            if (
                not isinstance(operand.type.memory_space, builtin.IntegerAttr)
                and operand.type.memory_space.value.data != 1
            ):
                return None

            # cast required: find previous cast or create new one
            cast_op = None
            for use in operand.uses:
                if (
                    isinstance(use.operation, memref.MemorySpaceCast)
                    and isinstance(use.operation.dest.type, memref.MemRefType)
                    and use.operation.dest.type.memory_space
                    == builtin.IntegerAttr(1, builtin.i32)
                ):
                    cast_op = use.operation
                    break
            # If cast op not found, create and insert new one
            if cast_op is None:
                cast_op = memref.MemorySpaceCast.from_type_and_target_space(
                    operand, operand.type, builtin.IntegerAttr(1, builtin.i32)
                )
                rewriter.insert_op_before_matched_op(cast_op)

            return cast_op

        # cast all inputs and outputs to correct memory space
        new_inputs = [
            inp if get_cast_op(inp) is None else get_cast_op(inp).dest
            for inp in op.inputs
        ]
        new_outputs = [
            out if get_cast_op(out) is None else get_cast_op(out).dest
            for out in op.outputs
        ]

        # new linalg op with new inputs & outputs
        linalg_op = linalg.Generic(
            new_inputs,
            new_outputs,
            rewriter.move_region_contents_to_new_regions(op.regions[0]),
            op.indexing_maps,
            op.iterator_types,
            op.doc,
            op.library_call,
        )

        # replace op
        rewriter.replace_matched_op(linalg_op)


class RealizeMemorySpaceCasts(RewritePattern):
    """Realize the inserted memory space casts. In the snitch
    cluster case, the different clusters can only access their own
    local TCDM L1 memory, so memory space casts are handled by local
    allocations and memref copies at the correct time.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.MemorySpaceCast, rewriter: PatternRewriter):
        # check if result is memreftype to make pyright happy
        if not isinstance(op.results[0].type, memref.MemRefType):
            return

        # create memref.dim operations for dynamic dimensions
        shapes = [x.value.data for x in op.results[0].type.shape.data]
        dyn_operands = []
        for i in range(len(shapes)):
            # Dynamic shapes are represented as -1
            if shapes[i] == -1:
                ## create dim op
                index = arith.Constant.from_int_and_width(i, builtin.IndexType())
                dim_op = memref.Dim.from_source_and_index(op.source, index.result)
                rewriter.insert_op_before_matched_op([index, dim_op])
                dyn_operands.append(dim_op)

        # replace cast with allocation
        alloc_op = memref.Alloc.get(
            op.results[0].type.get_element_type(),
            64,  # default 64 alignment (necessary ?)
            op.results[0].type.get_shape(),
            dynamic_sizes=dyn_operands,
            layout=op.results[0].type.layout,
            memory_space=op.results[0].type.memory_space,
        )

        # Insert copy ops if newly allocated memref is used as
        # input or output, list to visit all uses of allocated memrefs:
        uses = [x.operation for x in op.results[0].uses]

        # insert "copy to" for first use as input
        # walk parent op in order to find first use as input
        for use_op in op.parent.walk():
            if use_op not in uses:
                continue
            # check if input
            is_input = False
            if not isinstance(use_op, linalg.Generic):
                # don't know if input or output, default to yes
                is_input = True
            else:
                is_input = op.results[0] in use_op.inputs
            if is_input:
                # insert copy op
                copy_op = memref.CopyOp(op.source, op.dest)
                rewriter.insert_op_before(copy_op, use_op)
                break

        # insert "copy from" for last use as output
        # walk parent op in reverse order to find last use as output
        for use_op in op.parent.walk_reverse():
            if use_op not in uses:
                continue
            # check if input
            is_output = False
            if not isinstance(use_op, linalg.Generic):
                # don't know if input or output, default to yes
                is_output = True
            else:
                is_output = op.results[0] in use_op.outputs
            if is_output:
                # insert copy op
                copy_op = memref.CopyOp(op.dest, op.source)
                rewriter.insert_op_after(copy_op, use_op)
                break

        # finally, replace the casting operation with the allocation
        rewriter.replace_matched_op(alloc_op, new_results=[alloc_op.memref])


class SetMemorySpace(ModulePass):
    name = "set-memory-space"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(InitFuncMemorySpace()).rewrite_module(op)
        PatternRewriteWalker(InitLinalgMemorySpace()).rewrite_module(op)
        PatternRewriteWalker(RealizeMemorySpaceCasts()).rewrite_module(op)
