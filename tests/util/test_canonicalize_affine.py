from xdsl.ir.affine import (
    AffineConstantExpr,
    AffineDimExpr,
    AffineMap,
)

from snaxc.util.canonicalize_affine import canonicalize_expr, canonicalize_map


def test_canonicalize():
    c0 = AffineConstantExpr(0)
    c1 = AffineConstantExpr(1)
    c2 = AffineConstantExpr(2)
    c4 = AffineConstantExpr(4)

    d0 = AffineDimExpr(0)
    d1 = AffineDimExpr(1)
    d2 = AffineDimExpr(2)
    d3 = AffineDimExpr(3)

    # additions:
    # expr + 0
    assert canonicalize_expr(d0 + c0) == d0
    # 2 + 2
    assert canonicalize_expr(c2 + c2) == c4
    # c1 + expr
    assert canonicalize_expr(c1 + d0) == d0 + c1
    # d1 + d0
    assert canonicalize_expr(d1 + d0) == d0 + d1
    # (d1 * 4) + (d0 * 4)
    assert canonicalize_expr(d1 * c4 + d0 * c4) == d0 * c4 + d1 * c4

    # (a + b) + c
    assert canonicalize_expr((d1 + d2) + d3) == d1 + (d2 + d3)
    # ((a + b) + c) + d
    assert canonicalize_expr(((d0 + d1) + d2) + d3) == d0 + (d1 + (d2 + d3))

    # (a + b) * cst
    assert canonicalize_expr((d0 + d1) * c2) == (d0 * c2) + (d1 * c2)

    # multiplications:
    assert canonicalize_expr(c4 * d1) == d1 * c4

    # floordiv:
    # dividing by 1 is useless
    assert canonicalize_expr(d0 // c1) == d0

    # mod:
    # mod 1 is zero
    assert canonicalize_expr(d0 % c1) == c0

    # test map:
    assert canonicalize_map(AffineMap(2, 0, (c1 * d1, d0 + c0))) == AffineMap(
        2, 0, (d1, d0)
    )
