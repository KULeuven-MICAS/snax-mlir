from xdsl.dialects import builtin, func, memref
from xdsl.ir import MLContext
from xdsl.passes import ModulePass


class ClearMemorySpace(ModulePass):
    name = "clear-memory-space"

    def apply(self, ctx: MLContext, module: builtin.ModuleOp) -> None:
        # helper function to clear the memory space of a memref
        def clear_memory_space(t):
            if isinstance(t, memref.MemRefType):
                if not isinstance(t.memory_space, builtin.NoneAttr):
                    return memref.MemRefType(
                        t.element_type,
                        t.get_shape(),
                        t.layout,
                        builtin.NoneAttr(),
                    )
            return t

        for op_in_module in module.walk():
            for operand in op_in_module.operands:
                operand.type = clear_memory_space(operand.type)

            if isinstance(op_in_module, func.FuncOp):
                # special case for func ops because func ops do not have
                # operands, they have function_types which have ins & outs
                # Define new function type with updated inputs and outputs
                # mapped to a default memory space
                new_function_type = builtin.FunctionType.from_lists(
                    map(clear_memory_space, op_in_module.function_type.inputs),
                    map(clear_memory_space, op_in_module.function_type.outputs),
                )

                op_in_module.function_type = new_function_type
