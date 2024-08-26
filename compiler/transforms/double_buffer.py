from collections.abc import Iterator

from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, linalg, memref, scf
from xdsl.dialects.builtin import IndexType, NoneAttr
from xdsl.ir import Block, ErasedSSAValue, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from compiler.dialects import snax


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
        return False


def operates_on_copied_in(op: Operation, copy_in_ops: list[memref.CopyOp]) -> bool:
    """
    Check if the operation operates on the copied in memory
    """
    for copy_in_op in copy_in_ops:
        for input in op.operands:
            if copy_in_op.destination == input:
                return True
    return False


def operates_on_copied_out(op: Operation, copy_out_ops: list[memref.CopyOp]) -> bool:
    """
    Check if the operation operates on the copied in memory
    """
    for copy_out_op in copy_out_ops:
        for input in op.operands:
            if copy_out_op.source == input:
                return True
    return False


def can_double_buffer(for_op: scf.For) -> bool:
    """Check if the loop can be double buffered"""
    if get_int(for_op.step) is None:
        return False
    if get_int(for_op.ub) is None:
        return False
    if get_int(for_op.step) == get_int(for_op.ub):
        return False
    return True


def get_int(value: SSAValue) -> int:
    """
    Get the integer value of a SSAValue
    """
    if isinstance(value.owner, arith.Constant):
        return value.owner.value.value.data
    if isinstance(value.owner, arith.Addi):
        return get_int(value.owner.lhs) + get_int(value.owner.rhs)
    if isinstance(value.owner, arith.Subi):
        return get_int(value.owner.lhs) - get_int(value.owner.rhs)
    if isinstance(value.owner, arith.Muli):
        return get_int(value.owner.lhs) * get_int(value.owner.rhs)


def is_uneven_func(for_op: scf.For) -> bool:
    """
    Check if the loop is uneven
    """
    return get_int(for_op.ub) % (2 * get_int(for_op.step)) <= get_int(for_op.step)


def outside_loop(dependant_op: Operation, loop: scf.For) -> bool:
    """
    Check if the operation is outside of the loop
    """
    if dependant_op.parent_op() is loop:
        return False
    if dependant_op.parent is None:
        return True
    return outside_loop(dependant_op.parent, loop)


def replace_iter_value(
    dependant_op: Operation, loop: scf.For, old_value: SSAValue, new_value: SSAValue
):
    """
    replace the iteration value in the operation with the new value
    """
    if isinstance(dependant_op, Block):
        return
    if outside_loop(dependant_op, loop):
        return
    for index, operand in enumerate(dependant_op.operands):
        if operand == old_value:
            dependant_op.operands[index] = new_value
        else:
            replace_iter_value(operand.owner, loop, old_value, new_value)


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
        if not can_double_buffer(for_op):
            return
        is_uneven = is_uneven_func(for_op)
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

        # If either Copy In or Copy Out are empty, return
        if not copy_in_ops or not copy_out_ops:
            return

        # Duplicate the memref allocs
        inputs_with_clones = []
        for copy_in_op in copy_in_ops:
            alloc: memref.Alloc = copy_in_op.destination.owner
            # Buffer can't be duplicated when not found
            if not isinstance(alloc, memref.Alloc):
                return
            alloc_clone = alloc.clone()
            rewriter.insert_op(alloc_clone, InsertPoint.after(alloc))
            inputs_with_clones.append((alloc.results[0], alloc_clone.results[0]))

        outputs_with_clones = []
        for copy_out_op in copy_out_ops:
            alloc: memref.Alloc = copy_out_op.source.owner
            # Buffer can't be duplicated when not found
            if not isinstance(alloc, memref.Alloc):
                return
            alloc_clone = alloc.clone()
            rewriter.insert_op(alloc_clone, InsertPoint.after(alloc))
            outputs_with_clones.append((alloc.results[0], alloc_clone.results[0]))

        # Add a synchronization operation at the end of the loop
        # These will be used at the end of all logical blocks
        # insert before scf.yield
        sync_op = snax.ClusterSyncOp()
        rewriter.insert_op(sync_op, InsertPoint.before(for_op.body.blocks[0].last_op))

        # Duplicate the content of the loop
        for_copy = for_op.clone()

        # Replace in the original loop all computes with the cloned inputs and outputs
        for op in for_op.body.walk():
            if not isinstance(op, memref.CopyOp):
                if operates_on_copied_in(op, copy_in_ops):
                    for index, input_with_clone in enumerate(inputs_with_clones):
                        for operand_index, operand in enumerate(op.operands):
                            if operand == input_with_clone[0]:
                                op.operands[operand_index] = input_with_clone[1]
                if operates_on_copied_out(op, copy_out_ops):
                    for index, output_with_clone in enumerate(outputs_with_clones):
                        for result_index, result in enumerate(op.operands):
                            if result == output_with_clone[0]:
                                op.operands[result_index] = output_with_clone[1]

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
        initial_for_op = for_op.clone()
        initial_copy_in_ops = initial_for_op.body
        # Remove all compute operations and copy out operations
        lower_bound = for_op.lb
        local_copy_in_ops = []
        for op in initial_copy_in_ops.walk():
            if isinstance(op, memref.CopyOp) and (
                has_memory_space(op.destination, "L1")
                or has_memory_space(op.source, "L3")
            ):
                local_copy_in_ops.append(op)
                replace_iter_value(
                    op.source.owner,
                    initial_for_op,
                    initial_copy_in_ops.blocks[0].args[0],
                    lower_bound,
                )
            elif isinstance(op, memref.CopyOp) and (
                has_memory_space(op.source, "L1")
                or has_memory_space(op.destination, "L3")
            ):
                rewriter.erase_op(op)
            elif isinstance(op, linalg.Generic):
                rewriter.erase_op(op)
        # Replace the iteration value in the copy with static values
        for op in initial_copy_in_ops.walk():
            for index, operand in enumerate(op.operands):
                if operand == initial_copy_in_ops.blocks[0].args[0]:
                    op.operands[index] = lower_bound
        # Add the copy in front of the loop
        for op in soft_walk_region(initial_copy_in_ops):
            op.detach()
            if not isinstance(op, scf.Yield):
                rewriter.insert_op(op, InsertPoint.before(for_op))

        # Add second iteration of copy in
        second_clone_for = for_copy.clone()
        second_copy_in_ops = second_clone_for.body
        # Remove all copy out operations
        one_more_than_lower_bound = for_op.step
        for op in second_copy_in_ops.walk():
            if isinstance(op, memref.CopyOp) and (
                has_memory_space(op.destination, "L1")
                or has_memory_space(op.source, "L3")
            ):
                replace_iter_value(
                    op.source.owner,
                    second_clone_for,
                    second_copy_in_ops.blocks[0].args[0],
                    one_more_than_lower_bound,
                )
            if isinstance(op, linalg.Generic):
                replace_iter_value(
                    op,
                    second_clone_for,
                    second_copy_in_ops.blocks[0].args[0],
                    lower_bound,
                )
            if isinstance(op, memref.CopyOp) and has_memory_space(op.source, "L1"):
                rewriter.erase_op(op)
        # Replace the iteration value in the copy with static values
        for op in second_copy_in_ops.walk():
            for index, operand in enumerate(op.operands):
                if operand == second_copy_in_ops.blocks[0].args[0]:
                    op.operands[index] = one_more_than_lower_bound
        # Add the copy in front of the loop
        for op in soft_walk_region(second_copy_in_ops):
            op.detach()
            if not isinstance(op, scf.Yield):
                rewriter.insert_op(op, InsertPoint.before(for_op))

        # Add final copy out outside of the loop
        # Check unevenness to know which for loop to clone
        if is_uneven:
            final_copy_for = for_op.clone()
        else:
            final_copy_for = for_copy.clone()
        final_copy_out_ops = final_copy_for.body
        # Remove all copy in and compute operations
        one_less_than_upper_bound_op = arith.Subi(for_op.ub, for_op.step, IndexType())
        rewriter.insert_op(one_less_than_upper_bound_op, InsertPoint.before(for_op))
        local_copy_out_ops = []
        for op in final_copy_out_ops.walk(reverse=False):
            if isinstance(op, memref.CopyOp) and (
                has_memory_space(op.source, "L1")
                or has_memory_space(op.destination, "L3")
            ):
                local_copy_out_ops.append(op)
                replace_iter_value(
                    op.destination.owner,
                    final_copy_for,
                    final_copy_out_ops.blocks[0].args[0],
                    one_less_than_upper_bound_op.results[0],
                )
            elif isinstance(op, memref.CopyOp) and (
                has_memory_space(op.destination, "L1")
                or has_memory_space(op.source, "L3")
            ):
                rewriter.erase_op(op)
            elif isinstance(op, linalg.Generic):
                rewriter.erase_op(op)
        # delete all remaining uses of the iteration value
        for op in final_copy_out_ops.walk():
            for index, operand in enumerate(op.operands):
                if operand == final_copy_out_ops.blocks[0].args[0] or isinstance(
                    operand, ErasedSSAValue
                ):
                    rewriter.erase_op(op, safe_erase=False)
        # Add the copy after the loop
        for op in soft_walk_region(final_copy_out_ops, reverse=True):
            op.detach()
            if not isinstance(op, scf.Yield):
                rewriter.insert_op(op, InsertPoint.after(for_op))

        # Add second to last iteration of compute and copy out
        # Check unevenness to know which for loop to clone
        if is_uneven:
            second_to_last_for = for_copy.clone()
        else:
            second_to_last_for = for_op.clone()
        second_to_last_compute_ops = second_to_last_for.body
        # Remove all copy in operations
        const_2 = arith.Constant.from_int_and_width(2, IndexType())
        two_step = arith.Muli(for_op.step, const_2.results[0])
        two_less_than_upper_bound_op = arith.Subi(
            for_op.ub, two_step.results[0], IndexType()
        )
        for op in second_to_last_compute_ops.walk(reverse=False):
            if isinstance(op, memref.CopyOp) and has_memory_space(op.destination, "L1"):
                rewriter.erase_op(op)
            elif isinstance(op, memref.CopyOp) and has_memory_space(op.source, "L1"):
                replace_iter_value(
                    op.destination.owner,
                    second_to_last_for,
                    second_to_last_compute_ops.blocks[0].args[0],
                    two_less_than_upper_bound_op.results[0],
                )
            elif isinstance(op, linalg.Generic):
                replace_iter_value(
                    op,
                    second_to_last_for,
                    second_to_last_compute_ops.blocks[0].args[0],
                    one_less_than_upper_bound_op.results[0],
                )
        # delete all remaining uses of the iteration value
        for op in second_to_last_compute_ops.walk():
            for index, operand in enumerate(op.operands):
                if operand == second_to_last_compute_ops.blocks[0].args[
                    0
                ] or isinstance(operand, ErasedSSAValue):
                    rewriter.erase_op(op, safe_erase=False)
        # Add the compute and copy out after the loop
        for op in soft_walk_region(second_to_last_compute_ops, reverse=True):
            op.detach()
            if not isinstance(op, scf.Yield):
                rewriter.insert_op(op, InsertPoint.after(for_op))

        # When the number of iterations is uneven, add the last iteration of the loop
        const_3 = arith.Constant.from_int_and_width(3, IndexType())
        three_step = arith.Muli(for_op.step, const_3.results[0])
        three_less_than_upper_bound_op = arith.Subi(
            for_op.ub, three_step.results[0], IndexType()
        )
        if is_uneven:
            last_complete_for = for_op.clone()
            last_complete_ops = last_complete_for.body
            for op in final_copy_out_ops.walk(reverse=False):
                if isinstance(op, memref.CopyOp) and (
                    has_memory_space(op.source, "L1")
                    or has_memory_space(op.destination, "L3")
                ):
                    replace_iter_value(
                        op.destination.owner,
                        last_complete_for,
                        last_complete_ops.blocks[0].args[0],
                        one_less_than_upper_bound_op.results[0],
                    )
                elif isinstance(op, memref.CopyOp) and (
                    has_memory_space(op.destination, "L1")
                    or has_memory_space(op.source, "L3")
                ):
                    replace_iter_value(
                        op.source.owner,
                        last_complete_for,
                        last_complete_ops.blocks[0].args[0],
                        three_less_than_upper_bound_op.results[0],
                    )
                elif isinstance(op, linalg.Generic):
                    replace_iter_value(
                        op,
                        last_complete_for,
                        last_complete_ops.blocks[0].args[0],
                        two_less_than_upper_bound_op.results[0],
                    )

            for op in soft_walk_region(second_to_last_compute_ops, reverse=True):
                # Add after the loop
                op.detach()
                if not isinstance(op, scf.Yield):
                    rewriter.insert_op(op, InsertPoint.after(for_op))

        # add all adapted iteration values before the loop
        rewriter.insert_op(const_2, InsertPoint.before(for_op))
        rewriter.insert_op(two_step, InsertPoint.before(for_op))
        rewriter.insert_op(two_less_than_upper_bound_op, InsertPoint.before(for_op))

        rewriter.insert_op(const_3, InsertPoint.before(for_op))
        rewriter.insert_op(three_step, InsertPoint.before(for_op))
        rewriter.insert_op(three_less_than_upper_bound_op, InsertPoint.before(for_op))

        added_iteration_value = arith.Addi(for_op.body.blocks[0].args[0], for_op.step)
        subtracted_iteration_value = arith.Subi(
            for_op.body.blocks[0].args[0], for_op.step, IndexType()
        )
        double_subtracted_iteration_value = arith.Subi(
            for_op.body.blocks[0].args[0], two_step.results[0], IndexType()
        )
        # Replace in the copied loop all iteration values with the correct new iteration values
        for op in for_copy.body.walk():
            if isinstance(op, memref.CopyOp) and has_memory_space(op.destination, "L1"):
                replace_iter_value(
                    op.source.owner,
                    for_copy,
                    for_copy.body.blocks[0].args[0],
                    added_iteration_value.results[0],
                )
            elif isinstance(op, memref.CopyOp) and has_memory_space(op.source, "L1"):
                replace_iter_value(
                    op.destination.owner,
                    for_copy,
                    for_copy.body.blocks[0].args[0],
                    subtracted_iteration_value.results[0],
                )
        for op in for_op.body.walk():
            if isinstance(op, memref.CopyOp) and has_memory_space(op.source, "L1"):
                replace_iter_value(
                    op.destination.owner,
                    for_op,
                    for_op.body.blocks[0].args[0],
                    double_subtracted_iteration_value.results[0],
                )

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
        for_op.operands[2] = two_step.results[0]

        # Replace in the cloned loop all iteration values with the one added to the original step size

        rewriter.insert_op(
            added_iteration_value, InsertPoint.at_start(for_op.body.blocks[0])
        )
        rewriter.insert_op(
            subtracted_iteration_value, InsertPoint.at_start(for_op.body.blocks[0])
        )
        rewriter.insert_op(
            double_subtracted_iteration_value,
            InsertPoint.at_start(for_op.body.blocks[0]),
        )


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
