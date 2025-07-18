from xdsl.context import Context
from xdsl.dialects import builtin, func, linalg, memref
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa

from snaxc.dialects import dart
from snaxc.util.snax_memory import L1, L3


class InitFuncMemorySpace(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        """Add a default "L3" memory space to memrefs used in the function
        that do not have a memory space specified yet"""

        # Function must be public
        if op.sym_visibility is not None and op.sym_visibility.data != "public":
            return

        # Function must have memref arguments with an undefined memory space
        if not any(
            [
                isinstance(x, builtin.MemRefType) and isinstance(x.memory_space, builtin.NoneAttr)
                for x in [*op.function_type.inputs, *op.function_type.outputs]
            ]
        ):
            return

        # Mapping function to assign default memory space "L3"
        def change_to_memory_space(t: Attribute) -> Attribute:
            if isa(t, builtin.MemRefType[Attribute]):
                if isinstance(t.memory_space, builtin.NoneAttr):
                    return builtin.MemRefType(
                        t.element_type,
                        t.get_shape(),
                        t.layout,
                        L3.attribute,
                    )
            return t

        # Define new function type with updated inputs and outputs
        # mapped to a default memory space
        new_function_type = builtin.FunctionType.from_lists(
            list(map(change_to_memory_space, op.function_type.inputs)),
            list(map(change_to_memory_space, op.function_type.outputs)),
        )

        # Change region of function to use new argument types
        block = op.body.blocks.first
        assert block is not None
        for arg in block.args:
            new_arg = block.insert_arg(change_to_memory_space(arg.type), arg.index)
            arg.replace_by(new_arg)
            block.erase_arg(arg)

        # Define new function op with new type and copy region contents
        new_op = func.FuncOp(
            op.sym_name.data,
            new_function_type,
            region=rewriter.move_region_contents_to_new_regions(op.regions[0]),
            visibility=op.sym_visibility,
        )

        # Replace function op
        rewriter.replace_matched_op(new_op)


class InitMemRefGlobalMemorySpace(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.GetGlobalOp, rewriter: PatternRewriter):
        # global variables should go in memory space L3
        memspace = op.memref.type.memory_space

        # If memory space is already L3, don't do anything
        if memspace == L3.attribute:
            return

        # otherwise, create new memref type with correct memory space
        new_memref_type = builtin.MemRefType(
            op.memref.type.element_type,
            op.memref.type.get_shape(),
            op.memref.type.layout,
            L3.attribute,
        )

        # create new get_global op
        new_op = memref.GetGlobalOp(op.name_.root_reference.data, new_memref_type)

        # replace op
        rewriter.replace_matched_op(new_op)


class InitMemRefAllocMemorySpace(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.AllocOp, rewriter: PatternRewriter):
        # allocs should go in memory space L1
        memspace = op.memref.type.memory_space

        if memspace == L1.attribute:
            # good, nothing left to do
            return

        # create new alloc op
        new_op = memref.AllocOp.get(
            op.memref.type.element_type,
            op.alignment,
            op.memref.type.get_shape(),
            dynamic_sizes=op.dynamic_sizes,
            layout=op.memref.type.layout,
            memory_space=L1.attribute,
        )

        # replace op
        rewriter.replace_matched_op(new_op, new_results=[new_op.memref])


class InitStreamAndLinalgMemorySpace(RewritePattern):
    """
    Convert all linalg.generics and stream.streaming region ops to only use L1
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp | dart.OperationOp, rewriter: PatternRewriter):
        operands_to_memory_cast = tuple(
            x
            for x in op.operands
            if isinstance(memref_type := x.type, builtin.MemRefType) and memref_type.memory_space != L1.attribute
        )

        if not operands_to_memory_cast:
            return

        def get_cast_op(operand: SSAValue) -> memref.MemorySpaceCastOp:
            # cast required: find previous cast or create new one
            cast_op = None
            for use in operand.uses:
                if (
                    isinstance(use.operation, memref.MemorySpaceCastOp)
                    and isinstance(use_type := use.operation.dest.type, builtin.MemRefType)
                    and use_type.memory_space == L1.attribute
                ):
                    cast_op = use.operation
                    break
            # If cast op not found, create and insert new one
            assert isa(optype := operand.type, builtin.MemRefType[Attribute])
            if cast_op is None:
                cast_op = memref.MemorySpaceCastOp.from_type_and_target_space(operand, optype, L1.attribute)
                rewriter.insert_op_before_matched_op(cast_op)

            return cast_op

        # insert memory cast for every value
        memory_cast_ops: dict[SSAValue, memref.MemorySpaceCastOp] = {}
        for operand in operands_to_memory_cast:
            memory_cast_ops[operand] = get_cast_op(operand)

        # replace all operands to casted values
        for i in range(len(op.operands)):
            if op.operands[i] in operands_to_memory_cast:
                op.operands[i] = memory_cast_ops[op.operands[i]].dest


class HandleFuncReturns(RewritePattern):
    """Function returns which return a memref object must be replaced
    such that the memref object is returned in the memory space specified
    in the function handle"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.ReturnOp, rewriter: PatternRewriter):
        # get function op
        assert isinstance(func_op := op.parent_op(), func.FuncOp)

        outputs = [*func_op.function_type.outputs]

        new_arguments: list[SSAValue | Operation] = []
        changes_made = False

        # all outputs must be in the correct memory space
        for i in range(len(outputs)):
            func_op_output = outputs[i]
            func_return_output = op.arguments[i]

            if not isa(func_op_output, builtin.MemRefType[Attribute]):
                new_arguments.append(func_return_output)
            elif not isa(
                func_return_output_type := func_return_output.type,
                builtin.MemRefType[Attribute],
            ):
                new_arguments.append(func_return_output)
            elif func_op_output.memory_space != func_return_output_type.memory_space:
                # create cast op
                cast_op = memref.MemorySpaceCastOp.from_type_and_target_space(
                    func_return_output,
                    func_return_output_type,
                    func_op_output.memory_space,
                )
                rewriter.insert_op_before_matched_op(cast_op)

                # replace return value with cast
                new_arguments.append(cast_op)
                changes_made = True
            else:
                new_arguments.append(func_return_output)

        if not changes_made:
            return

        # create new return op
        new_op = func.ReturnOp(*new_arguments)

        # replace op
        rewriter.replace_matched_op(new_op)


class UpdateSubviewOperations(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.SubviewOp, rewriter: PatternRewriter):
        assert isa(op.source.type, builtin.MemRefType[Attribute])

        if op.source.type.memory_space != op.result.type.memory_space:
            # create new subview op with correct memory space
            new_op = memref.SubviewOp(
                op.source,
                op.offsets,
                op.sizes,
                op.strides,
                op.static_offsets,
                op.static_sizes,
                op.static_strides,
                result_type=builtin.MemRefType(
                    op.result.type.element_type,
                    op.result.type.get_shape(),
                    op.result.type.layout,
                    op.source.type.memory_space,
                ),
            )
            rewriter.replace_matched_op(new_op)


class SetMemorySpace(ModulePass):
    name = "set-memory-space"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    InitFuncMemorySpace(),
                    InitMemRefGlobalMemorySpace(),
                    InitMemRefAllocMemorySpace(),
                    HandleFuncReturns(),
                    InitStreamAndLinalgMemorySpace(),
                    UpdateSubviewOperations(),
                ]
            )
        ).rewrite_module(op)
