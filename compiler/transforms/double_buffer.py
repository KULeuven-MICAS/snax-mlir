from collections.abc import Iterator

from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, memref, scf
from xdsl.dialects.builtin import IndexType, NoneAttr
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


def soft_walk_region(region: Region, reverse: bool = False) -> Iterator[Operation]:
    """
    Call a function on all operations contained in the region.
    soft_walk will not go into nested regions.
    """
    for block in reversed(region.blocks) if reverse else region.blocks:
        yield from soft_walk_block(block=block, reverse=reverse)


def soft_walk_block(block: Block, reverse: bool = False) -> Iterator[Operation]:
    """
    Call a function on all operations contained in the block.
    soft_walk will not go into nested regions.
    """
    yield from reversed(block.ops) if reverse else block.ops


def has_memory_space(value: SSAValue, attr: str) -> bool:
    """
    Check if the operation has the given attribute
    """
    if not isinstance(value.type.memory_space, NoneAttr):
        return value.type.memory_space.data == attr
    else:
        return False  # TODO: Fix this


def my_are_equal(obj1, obj2):
    # Check if both objects are of the same type
    if type(obj1) is not type(obj2):
        return False

    # If the objects are of a basic type (int, str, float, etc.), compare directly
    if isinstance(obj1, int | float | str | bool):
        return obj1 == obj2

    # If the objects are lists or tuples, compare element-wise
    if isinstance(obj1, list | tuple):
        if len(obj1) != len(obj2):
            return False
        return all(my_are_equal(i1, i2) for i1, i2 in zip(obj1, obj2))

    # If the objects are dictionaries, compare keys and values recursively
    if isinstance(obj1, dict):
        if obj1.keys() != obj2.keys():
            return False
        return all(my_are_equal(obj1[k], obj2[k]) for k in obj1)

    # For other objects, recursively compare their __dict__ attributes
    if hasattr(obj1, "__dict__") and hasattr(obj2, "__dict__"):
        return my_are_equal(obj1.__dict__, obj2.__dict__)

    # If none of the above conditions are met, fall back to direct comparison
    return obj1 == obj2


def operates_on_copied_in(op: Operation, copy_in_ops: list[memref.CopyOp]) -> bool:
    """
    Check if the operation operates on the copied in memory
    """
    for copy_in_op in copy_in_ops:
        for input in op.operands:
            if my_are_equal(copy_in_op.destination, input):
                return True
    return False


class AddDoubleBuffer(RewritePattern):
    """
    Replace loops of the form
        For
            Copy In (1)
            Compute (1)
            Copy Out (1)
        end

    with

        Copy In (1)
        -----------
        Copy In (2)
        Compute (1)
        -----------
        For
            Copy In (1)
            Compute (2)
            Copy Out (1)
            ------------
            Copy In (2)
            Compute (1)
            Copy Out (2)
            ------------
        end
        Compute (2)
        Copy Out (1)
        -----------
        Copy Out (2)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, for_op: scf.For, rewriter: PatternRewriter):
        # Get all the Copy In operations
        copy_in_ops = []
        for op in soft_walk_region(for_op.body):
            if isinstance(op, memref.CopyOp):
                if has_memory_space(op.source, "L3") or has_memory_space(
                    op.destination, "L1"
                ):
                    copy_in_ops.append(op)
        # Get all the Copy Out operations
        copy_out_ops = []
        for op in soft_walk_region(for_op.body):
            if isinstance(op, memref.CopyOp):
                if has_memory_space(op.source, "L1") or has_memory_space(
                    op.destination, "L3"
                ):
                    copy_out_ops.append(op)

        # If either Copy In, Compute or Copy Out are empty, return
        if not copy_in_ops or not copy_out_ops:
            return

        # Duplicate the memref allocs
        inputs_with_clones = []
        for copy_in_op in copy_in_ops:
            alloc: memref.Alloc = copy_in_op.destination.owner
            alloc_clone = alloc.clone()
            rewriter.insert_op(alloc_clone, InsertPoint.after(alloc))
            inputs_with_clones.append((alloc.results[0], alloc_clone.results[0]))

        outputs_with_clones = []
        for copy_out_op in copy_out_ops:
            alloc: memref.Alloc = copy_out_op.source.owner
            alloc_clone = alloc.clone()
            rewriter.insert_op(alloc_clone, InsertPoint.after(alloc))
            outputs_with_clones.append((alloc.results[0], alloc_clone.results[0]))

        # Duplicate the content of the loop
        for_copy = for_op.clone()

        # Replace in the original loop all computes with the cloned inputs and outputs
        for op in for_op.body.walk():
            if operates_on_copied_in(op, copy_in_ops):
                for index, input_with_clone in enumerate(inputs_with_clones):
                    for operand_index, operand in enumerate(op.operands):
                        if operand == input_with_clone[0]:
                            op.operands[operand_index] = input_with_clone[1]
                for index, output_with_clone in enumerate(outputs_with_clones):
                    for result_index, result in enumerate(op.results):
                        if result == output_with_clone[0]:
                            op.results[result_index] = output_with_clone[1]

        # Replace in the cloned loop all copy in and copy out operations with the cloned inputs and outputs
        for op in for_copy.body.walk():
            if isinstance(op, memref.CopyOp):
                for index, input_with_clone in enumerate(inputs_with_clones):
                    if op.destination == input_with_clone[0]:
                        new_copy_op = memref.CopyOp(
                            source=op.source, destination=input_with_clone[1]
                        )
                        rewriter.replace_op(op, new_copy_op)
                for index, output_with_clone in enumerate(outputs_with_clones):
                    if op.source == output_with_clone[0]:
                        new_copy_op = memref.CopyOp(
                            source=output_with_clone[1], destination=op.destination
                        )
                        rewriter.replace_op(op, new_copy_op)

        # Add initial copy in
        initial_copy_in_ops = for_op.clone().body
        # Remove all compute operations and copy out operations
        lower_bound = for_op.lb
        local_copy_in_ops = []
        for op in initial_copy_in_ops.walk():
            if isinstance(op, memref.CopyOp) and (
                has_memory_space(op.destination, "L1")
                or has_memory_space(op.source, "L3")
            ):
                local_copy_in_ops.append(op)
            elif isinstance(op, memref.CopyOp) and (
                has_memory_space(op.source, "L1")
                or has_memory_space(op.destination, "L3")
            ):
                rewriter.erase_op(op)
            elif operates_on_copied_in(
                op, local_copy_in_ops
            ):  # TODO: shouldnt operate on copy_in_ops but its clone
                rewriter.erase_op(op)
            else:
                for index, operand in enumerate(op.operands):
                    if operand == initial_copy_in_ops.blocks[0].args[0]:
                        op.operands[index] = lower_bound
        for op in soft_walk_region(initial_copy_in_ops):
            # Add in front of the loop
            op.detach()
            if not isinstance(op, scf.Yield):
                rewriter.insert_op(op, InsertPoint.before(for_op))

        # Add second iteration of copy in
        second_copy_in_ops = for_copy.clone().body
        # Remove all copy out operations
        one_more_than_lower_bound = for_op.step
        for op in second_copy_in_ops.walk():
            if isinstance(op, memref.CopyOp) and has_memory_space(op.source, "L1"):
                rewriter.erase_op(op)
            else:
                for index, operand in enumerate(op.operands):
                    if operand == second_copy_in_ops.blocks[0].args[0]:
                        op.operands[index] = one_more_than_lower_bound

        for op in soft_walk_region(second_copy_in_ops):
            # Add in front of the loop
            op.detach()
            if not isinstance(op, scf.Yield):
                rewriter.insert_op(op, InsertPoint.before(for_op))

        # Add final copy out outside of the loop
        final_copy_out_ops = for_op.clone().body
        # Remove all copy in and compute operations
        one_less_than_upper_bound_op = arith.MinUI(for_op.ub, for_op.step, IndexType())
        rewriter.insert_op(one_less_than_upper_bound_op, InsertPoint.before(for_op))
        local_copy_out_ops = []
        for op in final_copy_out_ops.walk(reverse=False):
            if isinstance(op, memref.CopyOp) and (
                has_memory_space(op.source, "L1")
                or has_memory_space(op.destination, "L3")
            ):
                local_copy_out_ops.append(op)
            elif isinstance(op, memref.CopyOp) and (
                has_memory_space(op.destination, "L1")
                or has_memory_space(op.source, "L3")
            ):
                rewriter.erase_op(op)
            elif operates_on_copied_in(op, local_copy_out_ops):
                rewriter.erase_op(op)
            else:
                for index, operand in enumerate(op.operands):
                    if operand == final_copy_out_ops.blocks[0].args[0]:
                        op.operands[index] = one_less_than_upper_bound_op.results[0]
        for op in soft_walk_region(final_copy_out_ops, reverse=True):
            # Add in front of the loop
            op.detach()
            if not isinstance(op, scf.Yield):
                rewriter.insert_op(op, InsertPoint.after(for_op))

        # Add second to last iteration of compute and copy out
        second_to_last_compute_ops = for_copy.clone().body
        # Remove all copy in operations
        const_2 = arith.Constant.from_int_and_width(2, IndexType())
        two_step = arith.Muli(for_op.step, const_2.results[0])
        two_less_than_upper_bound_op = arith.MinUI(
            for_op.ub, two_step.results[0], IndexType()
        )
        for op in second_to_last_compute_ops.walk(reverse=False):
            if isinstance(op, memref.CopyOp) and has_memory_space(op.destination, "L1"):
                rewriter.erase_op(op)
            else:
                for index, operand in enumerate(op.operands):
                    if operand == second_to_last_compute_ops.blocks[0].args[0]:
                        op.operands[index] = two_less_than_upper_bound_op.results[0]

        for op in soft_walk_region(second_to_last_compute_ops, reverse=True):
            # Add in front of the loop
            op.detach()
            if not isinstance(op, scf.Yield):
                rewriter.insert_op(op, InsertPoint.after(for_op))

        rewriter.insert_op(const_2, InsertPoint.before(for_op))
        rewriter.insert_op(two_step, InsertPoint.before(for_op))
        rewriter.insert_op(two_less_than_upper_bound_op, InsertPoint.before(for_op))

        # Add the cloned loop to the original loop
        for_op.body.blocks[0].erase_op(for_op.body.blocks[0].ops.last)
        copy_cloned_ops = []
        for op in for_copy.body.blocks[0].ops:
            op.detach()
            for index, operand in enumerate(op.operands):
                if operand == for_copy.body.blocks[0].args[0]:
                    op.operands[index] = for_op.body.blocks[0].args[0]
            copy_cloned_ops.append(op)
        for_op.body.blocks[0].add_ops(copy_cloned_ops)

        # rewrite iteration step and bounds
        for_op.operands[0] = two_step.results[0]
        for_op.operands[1] = two_less_than_upper_bound_op.results[0]
        for_op.operands[2] = two_step.results[0]


class DoubleBuffer(ModulePass):
    """
    Transformation pass which applies double buffering to the module.
    Double buffering is possible when a loop exists with the following structure:
    'Copy In -> Compute -> Copy Out'.
    The pass will duplicate all dependencies and duplicate the content of the loop
    """

    name = "double-buffer"

    def apply(self, ctx: MLContext, module: builtin.ModuleOp) -> None:
        PatternRewriteWalker(AddDoubleBuffer(), apply_recursively=False).rewrite_module(
            module
        )
