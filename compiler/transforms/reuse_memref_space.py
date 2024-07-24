from xdsl.dialects import builtin, linalg, memref, arith, affine
from xdsl.ir import MLContext
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
        def can_move_alloc(op) -> bool:
            """
            Allocs can be moved if they comply with the following conditions:
            1. The operation is an Alloc.
            2. All alloc sizes are defined outside the loop or a maximum possible value can be derived.
            3. The resulting memref is not used outside the loop.
            """

            # Check if all operands are defined outside for loop or can be derived.
            def can_be_constant(expr, block) -> bool:
                if (
                    expr.parent is not block.walk
                ):  # TODO: Check if the expression is defined in the for-loop, Not sure if this is correct
                    return True
                if isinstance(expr, arith.Constant):
                    return True
                if isinstance(expr, affine.MinOp):
                    if can_be_constant(expr.VarOperand(0)):
                        return True
                    if can_be_constant(expr.VarOperand(1)):
                        return True
                return False

            # Check if the allocated memref is used outside the loop.
            # TODO: This is not correct, we need to check if the memref is used outside the loop.
            def used_outside_loop(op) -> bool:
                for use in op.uses:
                    if use.parent is not alloc_op.parent:
                        return True
                return False

            if all(
                isinstance(op, memref.Alloc),
                can_be_constant(op.dynamic_sizes(0), op.parent),
                can_be_constant(op.dynamic_sizes(1).op.parent),
                not used_outside_loop(op),
            ):
                return True
            return False

        # if the alloc can be moved, detach it from the parent and insert new alloc before the big for loop
        # TODO: Find the big for loop
        # When constants or maximal values where found from inside the loop, we need to insert these before the alloc.
        if can_move_alloc(alloc_op):
            alloc_op.detach()
            rewriter.insert_op(alloc_op, InsertPoint.before(big_for_op))


class ReuseMemrefSpace(ModulePass):
    name = "reuse-memref-space"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            MoveMemrefAllocations(), apply_recursively=False
        ).rewrite_module(op)
