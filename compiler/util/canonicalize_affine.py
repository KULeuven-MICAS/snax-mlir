from xdsl.ir.affine import (
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
    AffineConstantExpr,
    AffineDimExpr,
    AffineExpr,
    AffineMap,
)


def get_dim(expr: AffineExpr) -> int | None:
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
    assert expr.kind is AffineBinaryOpKind.Add
    if isinstance(expr.lhs, AffineConstantExpr):
        # lhs is constant
        if isinstance(expr.rhs, AffineConstantExpr):
            # both are constants
            return AffineConstantExpr(expr.lhs.value + expr.rhs.value)
        else:
            # move constant to rhs
            expr = AffineBinaryOpExpr(expr.kind, expr.rhs, expr.lhs)
    if isinstance(expr.rhs, AffineConstantExpr):
        # rhs is constant, lhs is not
        if expr.rhs.value == 0:
            return expr.lhs
    # order by minimum dimension first
    dim_lhs = get_dim(expr.lhs)  # 3
    dim_rhs = get_dim(expr.rhs)  # 0
    if dim_rhs is not None:
        if dim_lhs is None or dim_lhs > dim_rhs:
            return expr.rhs + expr.lhs
    return expr


def canonicalize_multiplication(expr: AffineBinaryOpExpr) -> AffineExpr:
    assert expr.kind is AffineBinaryOpKind.Mul
    if isinstance(expr.lhs, AffineConstantExpr):
        # lhs is constant
        if isinstance(expr.rhs, AffineConstantExpr):
            # both are constants
            return AffineConstantExpr(expr.lhs.value * expr.rhs.value)
        else:
            # move constant to rhs
            return AffineBinaryOpExpr(expr.kind, expr.rhs, expr.lhs)
    if isinstance(expr.rhs, AffineConstantExpr):
        # rhs is constant, lhs is not
        if expr.rhs.value == 1:
            return expr.lhs
    return expr


def canonicalize_floordiv(expr: AffineBinaryOpExpr) -> AffineExpr:
    assert expr.kind is AffineBinaryOpKind.FloorDiv
    if isinstance(expr.rhs, AffineConstantExpr):
        # rhs is constant, lhs is not
        if expr.rhs.value == 1:
            return expr.lhs
    return expr


def canonicalize_mod(expr: AffineBinaryOpExpr) -> AffineExpr:
    assert expr.kind is AffineBinaryOpKind.Mod
    if isinstance(expr.rhs, AffineConstantExpr):
        # rhs is constant, lhs is not
        if expr.rhs.value == 1:
            return AffineConstantExpr(0)
    return expr


def canonicalize_binary_op(expr: AffineBinaryOpExpr) -> AffineExpr:
    # canonicalize childern
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
    # canonicalize each result and construct new affine map
    return AffineMap(
        map.num_dims,
        map.num_symbols,
        tuple(canonicalize_expr(expr) for expr in map.results),
    )
