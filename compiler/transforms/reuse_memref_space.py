from xdsl.context import MLContext
from xdsl.dialects import affine, arith, builtin, memref, scf
from xdsl.dialects.builtin import IndexType
from xdsl.ir import Block, Operation, SSAValue
from xdsl.ir.affine import AffineConstantExpr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.traits import Pure


def find_parent_for_loop(op: Operation) -> scf.For | None:
    """
    Find the parent for-loop of the operation.
    If no for-loop is found, return None.
    """
    op = op.parent_op()
    if op is None:
        return None
    while not isinstance(op, scf.For):
        if op.parent_op() is None:
            return None
        op = op.parent_op()
    return op


def is_in_loop(op: Operation) -> bool:
    """
    Check if the operation is inside a loop.
    """
    return find_parent_for_loop(op) is not None


def defined_outside_loop(op: Operation) -> bool:
    """
    Check if all operands of the operation are defined outside the loop.
    """
    for operand in op.operands:
        if find_parent_for_loop(operand.owner) is find_parent_for_loop(op):
            return False
    return True


class LowerPureOperations(RewritePattern):
    """
    This class represents a rewrite pattern for moving any Operation outside
    a (nested) for-loop. An Oparation can be moved if it is pure, that is, it
    does not have any side-effects within the loop and is not dependent on any loop variable.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, main_op: Operation, rewriter: PatternRewriter):
        def can_move_operation(op: Operation) -> bool:
            """
            Operations can be moved if they comply with the following conditions:
            1. The operation is inside a loop.
            2. All operands are defined outside the loop.
            3. The operation is Pure, it doesnt have any side effects.
            """
            if all(
                [
                    is_in_loop(op),
                    defined_outside_loop(op),
                    Pure() in op.traits or isinstance(op, memref.Alloc),
                    not isinstance(op, scf.Yield),
                ]
            ):
                return True
            return False

        # if the alloc can be moved, detach it from the parent and insert new Alloc Object in front of the for-loop
        # When constants or maximal values where found from inside the loop, we need to insert these before the alloc.
        # Only the first for-loop is considered, the algorithm can be repeated to elevate the alloc-op higher.
        if can_move_operation(main_op):
            for_op = find_parent_for_loop(main_op)
            main_op.detach()

            rewriter.insert_op(main_op, InsertPoint.before(for_op))


class MoveMemrefDims(RewritePattern):
    """
    This class represents a rewrite pattern for moving Dim operations outside
    a (nested) for-loop. This is possible the dimensions of a memref are not dependent
    on the loop variables, or a maximum possible value can be derived.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, dim_op: memref.Dim, rewriter: PatternRewriter):
        def before_loop(op: Operation) -> bool:
            """
            Check if the operation is defined before the loop.
            """
            return find_parent_for_loop(op) is not find_parent_for_loop(dim_op)

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
            return False

        def dimension_outside_loop(dim_op: memref.Dim) -> bool:
            """
            Check if the dimension is defined outside the loop.
            """
            if before_loop(dim_op):
                return True
            memref_op = dim_op.source.owner
            if isinstance(memref_op, Block):
                # This happens when the dim is called on an input argument
                return True
            if isinstance(memref_op, memref.Subview):
                index = int(dim_op.index.owner.value.value.data)
                new_op = memref_op.sizes[index].owner
                if isinstance(new_op, arith.Constant) or isinstance(
                    new_op, affine.MinOp
                ):
                    return True
                elif isinstance(new_op, memref.Dim):
                    return dimension_outside_loop(new_op)
                else:
                    return False
            # TODO: Add support for other memref operations, like matmul
            return False

        def can_move_dim(dim_op: memref.Dim) -> bool:
            """
            Allocs can be moved if they comply with the following conditions:
            1.  The operation is a Dim operation.
            2.  The operation is inside a loop.
            3.  The memref uppon which the dimension is taken is defined outside the loop,
                or the asked size can be determined outside the loop.
            4.  The index is given by a constant operation, which will already be outside the loop.
            """
            pass
            if all(
                [
                    isinstance(dim_op, memref.Dim),
                    is_in_loop(dim_op),
                    isinstance(dim_op.index.owner, arith.Constant),
                ]
            ):
                return dimension_outside_loop(dim_op)
            return False

        def get_constant_value_from_other_constant(expr) -> arith.Constant:
            """
            Returns the constant value of the expression.
            """
            if isinstance(expr, int):
                return arith.Constant.from_int_and_width(expr, IndexType())
            if isinstance(expr, arith.Constant):
                return expr

        def get_constant_value_from_affine_min(expr: affine.MinOp) -> arith.Constant:
            """
            Returns the constant value of the expression.
            """
            if can_be_constant(expr.map.data.results[0]) and not can_be_constant(
                expr.map.data.results[0]
            ):
                return get_constant_value_from_other_constant(
                    expr.map.data.results[0].value
                )
            if can_be_constant(expr.map.data.results[0]):
                return get_constant_value_from_other_constant(
                    expr.map.data.results[0].value
                )

        def get_new_dim_op(
            dim_op: memref.Dim,
        ) -> memref.Dim | arith.Constant | affine.MinOp:
            """
            Returns the operation out of which the size can be determined outside the loop
            """
            if before_loop(dim_op):
                return dim_op
            memref_op = dim_op.source.owner
            if isinstance(memref_op, Block):
                # This happens when the dim is called on an input argument
                return dim_op
            if isinstance(memref_op, memref.Subview):
                index = int(dim_op.index.owner.value.value.data)
                new_op = memref_op.sizes[index].owner
                if isinstance(new_op, arith.Constant) or isinstance(
                    new_op, affine.MinOp
                ):
                    return new_op
                else:
                    return get_new_dim_op(new_op)
            # TODO: Add support for other memref operations, like matmul
            AssertionError("This Dim Operation is not replaceable")

        # if the alloc can be moved, detach it from the parent and insert new Alloc Object in front of the for-loop
        # When constants or maximal values where found from inside the loop, we need to insert these before the alloc.
        # Only the first for-loop is considered, the algorithm can be repeated to elevate the alloc-op higher.
        if can_move_dim(dim_op):
            new_dim_op = get_new_dim_op(dim_op)
            if isinstance(new_dim_op, affine.MinOp):
                temp_dim_op = get_constant_value_from_affine_min(new_dim_op)
                rewriter._replace_all_uses_with(
                    new_dim_op.results[0],
                    temp_dim_op.results[0],
                )
                new_dim_op = temp_dim_op
            if new_dim_op is not dim_op:
                rewriter._replace_all_uses_with(
                    # This is a private function of Rewriter Class and should be replaced with a public function,
                    # but for now it works
                    dim_op.results[0],
                    new_dim_op.results[0],
                )
            for_op = find_parent_for_loop(dim_op)

            if is_in_loop(new_dim_op):
                new_dim_op.detach()

            if new_dim_op.parent_op() is None:
                rewriter.insert_op(new_dim_op, InsertPoint.before(for_op))
            if new_dim_op is not dim_op:
                dim_op.detach()
                dim_op.erase(safe_erase=True)


class ReuseMemrefSpace(ModulePass):
    name = "reuse-memref-space"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerPureOperations(),
                    MoveMemrefDims(),
                ]
            ),
            apply_recursively=True,
            # First elevate outside first for-loop, then move outside the second for-loop (and optionally more)
        ).rewrite_module(op)
