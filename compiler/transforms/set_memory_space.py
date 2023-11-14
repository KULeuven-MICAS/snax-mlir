from xdsl.dialects import func, builtin, memref
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
)


class AddMemorySpace(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        """Add a default L3 memory space to memrefs used in the function
        that do not have a memory space specified yet"""

        # Function must be public
        if op.sym_visibility.data != "public":
            return

        # Function must have memref arguments with an undefined memory space
        if not any(
            [
                isinstance(x, memref.MemRefType)
                and isinstance(x.memory_space, builtin.NoneAttr)
                for x in op.function_type.inputs
            ]
        ) or any(
            [
                isinstance(x, memref.MemRefType)
                and isinstance(x.memory_space, builtin.NoneAttr)
                for x in op.function_type.outputs
            ]
        ):
            return

        # Mapping functino to assign default memory space 0
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

        # Define new funcion type with updated inputs & outputs
        new_function_type = builtin.FunctionType.from_lists(
            map(change_to_memory_space, op.function_type.inputs),
            map(change_to_memory_space, op.function_type.outputs),
        )

        # Change region of function to use new argument types
        for arg in op.args:
            arg.type = change_to_memory_space(arg.type)

        # Define op with new function type and copy region contents
        new_op = func.FuncOp(
            op.sym_name.data,
            new_function_type,
            region=rewriter.move_region_contents_to_new_regions(op.regions[0]),
            visibility=op.sym_visibility,
        )

        # Replice function op
        rewriter.replace_matched_op(new_op)


class SetMemorySpace(ModulePass):

    """Add a default L3 memory space to memrefs used in the function
    that do not have a memory space specified yet"""

    name = "set-memory-space"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(AddMemorySpace()).rewrite_module(op)
