from xdsl.ir.affine import (
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
    AffineConstantExpr,
    AffineDimExpr,
    AffineExpr,
    AffineMap,
)


def get_dim(expr: AffineExpr) -> int | None:
    """
    Get the dimension of this expression. If d0 appears, return 0.
    If d1 appears, return 1. This will fail for nested additions in
    which multiple dimension expressions appear. This happens quite
    often, but can only be solved with a full flattening of the affine map.
    That is to do as future work.
    """
    if isinstance(expr, AffineDimExpr):
        return expr.position
    if isinstance(expr, AffineBinaryOpExpr):
        result = get_dim(expr.lhs)
        if result is not None:
            return result
        result = get_dim(expr.rhs)
        if result is not None:
            return result


def canonicalize_addition(expr: AffineBinaryOpExpr) -> AffineExpr:
    """
    Canonicalizes an addition by:
        putting the constant on rhs
        folding the op if both operands are constant
        omitting a + 0
        ordering the operands by their dimension
    """
    # always put the constant on rhs
    assert expr.kind is AffineBinaryOpKind.Add
    if isinstance(expr.lhs, AffineConstantExpr):
        # lhs is constant
        if isinstance(expr.rhs, AffineConstantExpr):
            # both are constants: fold constants
            return AffineConstantExpr(expr.lhs.value + expr.rhs.value)
        else:
            # move constant to rhs
            expr = AffineBinaryOpExpr(expr.kind, expr.rhs, expr.lhs)
    if isinstance(expr.rhs, AffineConstantExpr):
        # rhs is constant, lhs is not
        # addition by 0 can be omitted
        if expr.rhs.value == 0:
            return expr.lhs
    # order by minimum dimension first
    # d0 + d1: good. d1 + d0: bad
    dim_lhs = get_dim(expr.lhs)
    dim_rhs = get_dim(expr.rhs)
    if dim_rhs is not None:
        if dim_lhs is None or dim_lhs > dim_rhs:
            return expr.rhs + expr.lhs
    return expr


def canonicalize_multiplication(expr: AffineBinaryOpExpr) -> AffineExpr:
    """
    Canonicalizes a multiplication by:
        putting the constant on rhs
        folding the op if both operands are constant
        omitting a * 1
        ordering the operands by their dimension
    """
    # always put the constant on rhs
    assert expr.kind is AffineBinaryOpKind.Mul
    if isinstance(expr.lhs, AffineConstantExpr):
        # lhs is constant
        if isinstance(expr.rhs, AffineConstantExpr):
            # both are constants: fold constants
            return AffineConstantExpr(expr.lhs.value * expr.rhs.value)
        else:
            # move constant to rhs
            return AffineBinaryOpExpr(expr.kind, expr.rhs, expr.lhs)
    if isinstance(expr.rhs, AffineConstantExpr):
        # rhs is constant, lhs is not
        # multiplication by 1 can be omitted
        if expr.rhs.value == 1:
            return expr.lhs
    return expr


def canonicalize_floordiv(expr: AffineBinaryOpExpr) -> AffineExpr:
    """
    Canonicalizes a floordiv by:
        omitting a // 1
    """
    assert expr.kind is AffineBinaryOpKind.FloorDiv
    if isinstance(expr.rhs, AffineConstantExpr):
        # division by 1 can be omitted
        if expr.rhs.value == 1:
            return expr.lhs
    return expr


def canonicalize_mod(expr: AffineBinaryOpExpr) -> AffineExpr:
    """
    Canonicalizes a module operation by:
        replacing a % 1 by constant 0
    """
    assert expr.kind is AffineBinaryOpKind.Mod
    if isinstance(expr.rhs, AffineConstantExpr):
        # module 1 is always 0
        if expr.rhs.value == 1:
            return AffineConstantExpr(0)
    return expr


def canonicalize_binary_op(expr: AffineBinaryOpExpr) -> AffineExpr:
    expr = AffineBinaryOpExpr(
        expr.kind, canonicalize_expr(expr.lhs), canonicalize_expr(expr.rhs)
    )
    if expr.kind is AffineBinaryOpKind.Add:
        return canonicalize_addition(expr)
    if expr.kind is AffineBinaryOpKind.Mul:
        return canonicalize_multiplication(expr)
    if expr.kind is AffineBinaryOpKind.FloorDiv:
        return canonicalize_floordiv(expr)
    if expr.kind is AffineBinaryOpKind.Mod:
        return canonicalize_mod(expr)
    return expr


def canonicalize_expr(expr: AffineExpr) -> AffineExpr:
    if isinstance(expr, AffineBinaryOpExpr):
        return canonicalize_binary_op(expr)

    return expr


# helper function to canonicalize affine maps
def canonicalize_map(map: AffineMap) -> AffineMap:
    # canonicalize each result expression of the map and construct new affine map
    return AffineMap(
        map.num_dims,
        map.num_symbols,
        tuple(canonicalize_expr(expr) for expr in map.results),
    )
