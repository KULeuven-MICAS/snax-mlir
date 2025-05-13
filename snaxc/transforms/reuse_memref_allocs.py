from xdsl.context import Context
from xdsl.dialects import affine, arith, builtin, memref, scf
from xdsl.dialects.builtin import IndexType, IntegerAttr
from xdsl.ir import Block, Operation, SSAValue
from xdsl.ir.affine import AffineConstantExpr
from xdsl.parser import AffineExpr
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
from xdsl.utils.hints import isa


def find_parent_for_loop(op: Operation) -> scf.ForOp | None:
    """
    Find the parent for-loop of the operation.
    If no for-loop is found, return None.
    """
    if (parent_op := op.parent_op()) is None:
        return None
    while not isinstance(parent_op, scf.ForOp):
        if parent_op.parent_op() is None:
            return None
        parent_op = parent_op.parent_op()
        assert parent_op is not None
    return parent_op


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
        if isinstance(operand.owner, Block):
            return False
        elif find_parent_for_loop(operand.owner) is find_parent_for_loop(op):
            return False
    return True


class LoopHoistPureOperations(RewritePattern):
    """
    This class represents a rewrite pattern for moving any Operation outside
    a (nested) for-loop. An Operation can be moved if it is pure, that is, it
    does not have any side-effects within the loop and is not dependent on any loop variable.
    Alloc operations are a special case, as they are not pure, but can be moved as well for optimization purposes.
    """

    # Add operations that are allowed to be moved, even if they are not pure
    whitelisted_ops: list[type[Operation]]

    def __init__(self, whitelisted_ops: list[type[Operation]]) -> None:
        self.whitelisted_ops = whitelisted_ops.copy()
        super().__init__()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, main_op: Operation, rewriter: PatternRewriter):
        def is_whitelisted(op: Operation) -> bool:
            """
            Check if the operation is whitelisted.
            It can be moved even if it is not pure.
            """
            for op_type in self.whitelisted_ops:
                if isinstance(op, op_type):
                    return True
            return False

        def can_move_operation(op: Operation) -> bool:
            """
            Operations can be moved if they comply with the following conditions:
            1. The operation is inside a loop.
            2. All operands are defined outside the loop.
            3. The operation is Pure, it doesnt have any side effects, or it is an Alloc operation.
            """
            if all(
                [
                    is_in_loop(op),
                    defined_outside_loop(op),
                    Pure() in op.traits or is_whitelisted(main_op),
                    not isinstance(op, scf.YieldOp),
                ]
            ):
                return True
            return False

        # if the alloc can be moved, detach it from the parent and insert new Alloc Object in front of the for-loop
        # When constants or maximal values where found from inside the loop, we need to insert these before the alloc.
        # Only the first for-loop is considered, the algorithm can be repeated to elevate the alloc-op higher.
        if can_move_operation(main_op):
            for_op = find_parent_for_loop(main_op)
            assert for_op is not None
            main_op.detach()

            rewriter.insert_op(main_op, InsertPoint.before(for_op))


class MoveMemrefDims(RewritePattern):
    """
    This class represents a rewrite pattern for moving Dim operations outside
    a (nested) for-loop. This is possible if the dimensions of a memref are not dependent
    on the loop variables, or a maximum possible value can be derived.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, dim_op: memref.DimOp, rewriter: PatternRewriter):
        def before_loop(op: Operation) -> bool:
            """
            Check if the operation is defined before the loop.
            """
            return find_parent_for_loop(op) is not find_parent_for_loop(dim_op)

        def can_be_constant(val: int | AffineExpr | SSAValue) -> bool:
            """
            Check if all operands are defined outside for loop or can be derived.
            """

            if isinstance(val, int):
                return True
            if isinstance(val, AffineConstantExpr):
                return True
            elif isinstance(val, AffineExpr):
                return False

            # Use the owner of the value to check what kind of operation it is
            expr = val.owner
            assert isinstance(expr, Operation)
            if before_loop(expr):
                return True
            return False

        def dimension_outside_loop(dim_op: memref.DimOp) -> bool:
            """
            Check if the dimension can be derived outside of this for-loop.
            """
            if before_loop(dim_op):
                return True
            memref_op = dim_op.source.owner
            index = dim_op.index.owner
            assert isinstance(dim_op.index.owner, arith.ConstantOp)
            assert isa(dim_op.index.owner.value, IntegerAttr[IndexType])
            index = int(dim_op.index.owner.value.value.data)
            return memref_op_outside_loop(memref_op, index)

        def get_subview_dim(subview: memref.SubviewOp, index: int) -> int | SSAValue:
            """
            Returns the size of the subview at the given index,
            if the size is dynamic, the value is returned as an SSAValue
            """
            static_sizes = subview.static_sizes.get_values()
            target = static_sizes[index]
            if not target == memref.SubviewOp.DYNAMIC_INDEX:
                assert isinstance(target, int)
                return target
            else:
                # If the size is dynamic, it is retrieved as an operand,
                # indexed w.r.t. the total number of dynamic sizes.
                magic_numbers = 0
                for i in range(index):
                    if static_sizes[i] == memref.SubviewOp.DYNAMIC_INDEX:
                        magic_numbers += 1
                return subview.sizes[magic_numbers]

        def memref_op_outside_loop(memref_op: Operation | Block, index: int) -> bool:
            if isinstance(memref_op, Block):
                # This happens when the dim is called on an input argument
                return True
            if isinstance(memref_op, memref.SubviewOp):
                subview_size = get_subview_dim(memref_op, index)
                if isinstance(subview_size, int):
                    return True
                new_op = subview_size.owner
                if isinstance(new_op, arith.ConstantOp) or isinstance(
                    new_op, affine.MinOp
                ):
                    return True
                elif isinstance(new_op, memref.DimOp):
                    return dimension_outside_loop(new_op)
                else:
                    return False
            return False

        def used_by_neither_alloc_nor_subview(dim_op: memref.DimOp) -> bool:
            """
            Check if the Dim operation is used by an operation that is not an Alloc or Subview operation.
            """
            for use in dim_op.results[0].uses:
                if not isinstance(use.operation, memref.AllocOp) and not isinstance(
                    use.operation, memref.SubviewOp
                ):
                    return True
            return False

        def can_move_dim(dim_op: memref.DimOp) -> bool:
            """
            Dims can be moved if they comply with the following conditions:
            1.  The operation is a Dim operation.
            2.  The operation is inside a loop.
            3.  The memref of which the dimension is taken is defined outside the loop,
                or the asked size can be determined outside the loop.
            4.  The index is given by a constant operation, which will already be outside the loop.
            5.  The Dim result is only used by Alloc operations and subview operations.
            """
            if all(
                [
                    is_in_loop(dim_op),
                    isinstance(dim_op.index.owner, arith.ConstantOp),
                    not used_by_neither_alloc_nor_subview(dim_op),
                ]
            ):
                return dimension_outside_loop(dim_op)
            return False

        def get_constant_value_from_other_constant(
            expr: int | arith.ConstantOp,
        ) -> arith.ConstantOp:
            """
            Returns the constant value of the expression.
            """
            if isinstance(expr, int):
                return arith.ConstantOp.from_int_and_width(expr, IndexType())
            else:
                return expr

        def get_constant_value_from_affine_min(expr: affine.MinOp) -> arith.ConstantOp:
            """
            Returns the constant value of the expression.
            """
            if can_be_constant(expr.map.data.results[0]) and not can_be_constant(
                expr.map.data.results[0]
            ):
                assert isinstance(expr.map.data.results[0], AffineConstantExpr)
                return get_constant_value_from_other_constant(
                    expr.map.data.results[0].value
                )
            if can_be_constant(expr.map.data.results[0]):
                assert isinstance(expr.map.data.results[0], AffineConstantExpr)
                return get_constant_value_from_other_constant(
                    expr.map.data.results[0].value
                )
            raise RuntimeError("no constant value found")

        def get_new_dim_op(
            dim_op: memref.DimOp,
        ) -> memref.DimOp | arith.ConstantOp | affine.MinOp:
            """
            Returns the operation out of which the size can be determined outside the loop
            """
            if before_loop(dim_op):
                return dim_op
            memref_ssa = dim_op.source
            index_constant = dim_op.index.owner
            assert isinstance(index_constant, arith.ConstantOp)
            return get_new_memref_op(memref_ssa, index_constant)

        def get_new_memref_op(
            memref_ssa: SSAValue, index_constant: arith.ConstantOp
        ) -> memref.DimOp | arith.ConstantOp | affine.MinOp:
            memref_op = memref_ssa.owner
            assert isa(index_constant.value, IntegerAttr[IndexType])
            index = index_constant.value.value.data
            if isinstance(memref_op, Block):
                # This happens when the dim is called on an input argument
                return memref.DimOp.from_source_and_index(memref_ssa, index_constant)
            if isinstance(memref_op, memref.SubviewOp):
                subview_size = get_subview_dim(memref_op, index)
                if isinstance(subview_size, int):
                    return arith.ConstantOp.from_int_and_width(
                        subview_size, IndexType()
                    )
                new_op = subview_size.owner
                if isinstance(new_op, arith.ConstantOp) or isinstance(
                    new_op, affine.MinOp
                ):
                    return new_op
                else:
                    assert isinstance(new_op, memref.DimOp)
                    return get_new_dim_op(new_op)
            raise AssertionError("This Dim Operation is not replaceable")

        # if the alloc can be moved, detach it from the parent and insert new Alloc Object in front of the for-loop
        # When constants or maximal values where found from inside the loop, we need to insert these before the alloc.
        # Only the first for-loop is considered, the algorithm can be repeated to elevate the alloc-op higher.
        if can_move_dim(dim_op):
            new_dim_op = get_new_dim_op(dim_op)
            if isinstance(new_dim_op, affine.MinOp):
                temp_dim_op = get_constant_value_from_affine_min(new_dim_op)
                new_dim_op.results[0].replace_by(temp_dim_op.results[0])
                new_dim_op = temp_dim_op
            if new_dim_op is not dim_op:
                dim_op.results[0].replace_by(new_dim_op.results[0])
            for_op = find_parent_for_loop(dim_op)

            if is_in_loop(new_dim_op):
                new_dim_op.detach()

            assert for_op is not None
            if new_dim_op.parent_op() is None:
                rewriter.insert_op(new_dim_op, InsertPoint.before(for_op))
            if new_dim_op is not dim_op:
                dim_op.detach()
                dim_op.erase(safe_erase=True)


class ReuseMemrefAllocs(ModulePass):
    name = "reuse-memref-allocs"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LoopHoistPureOperations([memref.AllocOp]),
                    MoveMemrefDims(),
                ]
            ),
            apply_recursively=True,
            # First elevate outside first for-loop, then move outside the second for-loop (and optionally more)
        ).rewrite_module(op)
