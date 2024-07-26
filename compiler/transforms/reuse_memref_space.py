from collections.abc import Sequence

from xdsl.context import MLContext
from xdsl.dialects import affine, arith, builtin, memref, scf
from xdsl.dialects.builtin import IndexType
from xdsl.ir import Block, Operation, OpResult, SSAValue
from xdsl.ir.affine import AffineConstantExpr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


class MoveMemrefAllocations(RewritePattern):
    """
    This class represents a rewrite pattern for moving memref allocations outside
    a double for-loop. This is possible when each loop a new memref-space is allocated with
    identical sizes and the space is not used outside the loop.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, alloc_op: memref.Alloc, rewriter: PatternRewriter):
        def find_parent_for_loop(op: Operation) -> scf.For:
            """
            Find the parent for-loop of the operation.
            If no for-loop is found, return None.
            """
            op = op.parent_op()
            while not isinstance(op, scf.For):
                if op.parent_op() is None:
                    return None
                op = op.parent_op()
            return op

        def before_loop(op: Operation) -> bool:
            """
            Check if the operation is defined before the loop.
            """
            return find_parent_for_loop(op) is not find_parent_for_loop(alloc_op)

        def is_in_loop(op: Operation) -> bool:
            """
            Check if the operation is inside a loop.
            """
            return find_parent_for_loop(op) is not None

        def can_move_dim(dim: memref.Dim) -> bool:
            """
            Check if the Dim operation can be moved outside the loop.
            """
            if before_loop(dim):
                return True
            subview = dim.source.owner
            if isinstance(subview, Block):
                # This happens when the dim is called on an input argument
                return True
            assert isinstance(subview, memref.Subview)
            index = int(dim.index.owner.value.value.data)
            new_op = subview.sizes[index].owner
            if isinstance(new_op, arith.Constant) or isinstance(new_op, affine.MinOp):
                return True
            elif isinstance(new_op, memref.Dim):
                return can_move_dim(new_op)
            else:
                return False

        def can_be_constant(val: SSAValue) -> bool:
            """
            Check if all operands are defined outside for loop or can be derived.
            """

            if isinstance(val, int):
                return True
            if isinstance(val, AffineConstantExpr):
                return True

            # Use the owner of the value to check what kind of operation it is
            expr = val.owner
            if before_loop(expr):
                return True
            if isinstance(expr, arith.Constant):
                return True
            if isinstance(expr, affine.MinOp):
                if can_be_constant(expr.map.data.results[0]):
                    return True
                if can_be_constant(expr.map.data.results[1]):
                    return True
            if isinstance(expr, memref.Dim):
                return can_move_dim(expr)
            return False

        def used_outside_parent_for(use: OpResult) -> bool:
            """
            Check if the allocated memref is used outside the loop.
            checks all parent-for loops of the use operation.
            """
            current_op = use.operation
            parent_for = find_parent_for_loop(alloc_op)
            while current_op is not None:
                if current_op is parent_for:
                    return False
                current_op = find_parent_for_loop(current_op)
            return True

        def used_outside_loop(op_results: list[OpResult]) -> bool:
            """
            Check if the allocated memref is used outside the loop.
            """
            for op_result in op_results:
                for use in op_result.uses:
                    if used_outside_parent_for(use):
                        return True
            return False

        def can_move_alloc(op: Operation) -> bool:
            """
            Allocs can be moved if they comply with the following conditions:
            1. The operation is an Alloc.
            2. The operation is inside a loop.
            3. All alloc sizes are defined outside the loop or a maximum possible value can be derived.
            4. The resulting memref is not used outside the loop.
            """
            assert isinstance(op, Operation)
            if all(
                [
                    isinstance(op, memref.Alloc),
                    is_in_loop(op),
                    not used_outside_loop(op.results),
                ]
            ):
                for size in op.dynamic_sizes:
                    if not can_be_constant(size):
                        return False
                return True
            return False

        def get_constant_value(expr) -> arith.Constant:
            """
            Returns the constant value of the expression.
            """
            if isinstance(expr, int):
                return arith.Constant.from_int_and_width(expr, IndexType())
            if isinstance(expr, arith.Constant):
                return expr.clone_without_regions()
            # If the expression is a MinOp, we can extract the maximum constant value from the operands
            if isinstance(expr, affine.MinOp):
                if can_be_constant(expr.map.data.results[0]) and not can_be_constant(
                    expr.map.data.results[0]
                ):
                    return get_constant_value(expr.map.data.results[0].value)
                if can_be_constant(expr.map.data.results[0]):
                    return get_constant_value(expr.map.data.results[0].value)

        def replace_dim(dim: memref.Dim) -> Operation:
            """
            Replace Dim operation with either Dim outside the loop, or a new Constant.
            """
            if before_loop(dim):
                return dim
            subview = dim.source.owner
            if isinstance(subview, Block):
                # This happens when the dim is called on an input argument
                return dim
            assert isinstance(subview, memref.Subview)
            index = int(dim.index.owner.value.value.data)
            new_op = subview.sizes[index].owner
            if isinstance(new_op, arith.Constant) or isinstance(new_op, affine.MinOp):
                return get_constant_value(new_op)
            else:
                return replace_dim(new_op)

        def get_source_before_for_loop(memref_val: SSAValue) -> SSAValue:
            """
            Returns the source of the old source before the for-loop.
            """
            while not before_loop(memref_val.owner):
                memref_val = memref_val.owner.source
            return memref_val

        def get_dynamic_sizes_and_add(alloc_op, ops_to_add) -> Sequence[SSAValue]:
            """
            Returns the dynamic sizes of the alloc operation.
            If the size is defined before the loop, it is kept as is.
            If the size can be derived from a constant, the constant is created and added to ops_to_add.
            """
            dynamic_sizes = []
            for size in alloc_op.dynamic_sizes:
                assert isinstance(size.owner, Operation)
                if before_loop(size.owner):
                    dynamic_sizes.append(size)
                elif isinstance(size.owner, memref.Dim):
                    new_constant = size.owner.index.owner.clone_without_regions()
                    ops_to_add.append(new_constant)
                    source_op = replace_dim(size.owner)
                    if isinstance(source_op, arith.Constant):
                        dynamic_sizes.append(source_op.results[0])
                        ops_to_add.append(source_op)
                    else:
                        new_dim = memref.Dim.from_source_and_index(
                            get_source_before_for_loop(size.owner.source),
                            new_constant.results[0],
                        )
                        ops_to_add.append(new_dim)
                        dynamic_sizes.append(new_dim.results[0])
                else:
                    new_constant = get_constant_value(size.owner)
                    ops_to_add.append(new_constant)
                    dynamic_sizes.append(new_constant.results[0])
            return dynamic_sizes

        # Repeat When Changes are made.
        changes_made = False
        # if the alloc can be moved, detach it from the parent and insert new Alloc Object in front of the for-loop
        # When constants or maximal values where found from inside the loop, we need to insert these before the alloc.
        # Only the first for-loop is considered, the algorithm can be repeated to elevate the alloc-op higher.
        if can_move_alloc(alloc_op):
            changes_made = True
            ops_to_add = []
            new_alloc_op = memref.Alloc(
                dynamic_sizes=get_dynamic_sizes_and_add(alloc_op, ops_to_add),
                symbol_operands=alloc_op.symbol_operands,
                result_type=alloc_op.results[0].type,
                alignment=alloc_op.alignment,
            )
            rewriter._replace_all_uses_with(
                # This is a private function of Rewriter Class and should be replaced with a public function,
                # but for now it works
                alloc_op.results[0],
                new_alloc_op.results[0],
            )
            ops_to_add.append(new_alloc_op)

            for_op = find_parent_for_loop(alloc_op)
            alloc_op.detach()

            for op in ops_to_add:
                rewriter.insert_op(op, InsertPoint.before(for_op))
            alloc_op.erase(safe_erase=True)
        return changes_made


class ReuseMemrefSpace(ModulePass):
    name = "reuse-memref-space"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            MoveMemrefAllocations(),
            apply_recursively=True,
            # First elevate outside first for-loop, then move outside the second for-loop (and optionally more)
        ).rewrite_module(op)
