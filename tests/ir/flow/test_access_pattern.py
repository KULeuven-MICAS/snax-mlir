import pytest
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap

from compiler.ir.flow import AccessPattern, SchedulePattern, TemplatePattern


# Pytest tests
def test_access_pattern_creation():
    pattern = AffineMap(
        num_dims=3,
        num_symbols=0,
        results=(AffineDimExpr(0), AffineDimExpr(1), AffineDimExpr(2))
    )
    bounds = (10, 20, 30)
    access_pattern = AccessPattern(bounds, pattern)
    assert access_pattern.bounds == bounds
    assert access_pattern.pattern == pattern
    assert access_pattern.num_dims == 3

def test_access_pattern_rotate():

    pattern = AffineMap(
        num_dims=3,
        num_symbols=0,
        results=(AffineDimExpr(0), AffineDimExpr(1), AffineDimExpr(2))
    )
    bounds = (10, 20, 30)
    access_pattern = AccessPattern(bounds, pattern)


    # test 1: 3 dims, rotate 2
    rotated_pattern = access_pattern.rotate(2)
    expected_bounds = (20, 10, 30)
    expected_results=(AffineDimExpr(1), AffineDimExpr(0), AffineDimExpr(2))
    assert rotated_pattern.bounds == expected_bounds
    assert rotated_pattern.pattern.results == expected_results
    assert isinstance(rotated_pattern, AccessPattern)

    # test 2: 3 dims, rotate 3
    rotated_pattern = access_pattern.rotate(3)
    expected_bounds = (20, 30, 10)
    expected_results=(AffineDimExpr(1), AffineDimExpr(2), AffineDimExpr(0))
    assert rotated_pattern.bounds == expected_bounds
    assert rotated_pattern.pattern.results == expected_results
    assert isinstance(rotated_pattern, AccessPattern)

    # test 3: 3 dims, rotate 1
    rotated_pattern = access_pattern.rotate(1)
    expected_bounds = (10, 20, 30)
    expected_results=(AffineDimExpr(0), AffineDimExpr(1), AffineDimExpr(2))
    assert rotated_pattern.bounds == expected_bounds
    assert rotated_pattern.pattern.results == expected_results
    assert isinstance(rotated_pattern, AccessPattern)


def test_access_pattern_disable_dims():
    pattern = AffineMap(
        num_dims=3,
        num_symbols=0,
        results=(AffineDimExpr(0), AffineDimExpr(1), AffineDimExpr(2))
    )
    bounds = (10, 20, 30)
    access_pattern = AccessPattern(bounds, pattern)

    # test 1: disable 0 dims (none)
    disabled_pattern = access_pattern.disable_dims(0)
    expected_bounds = (10, 20, 30)
    expected_results = (AffineDimExpr(0), AffineDimExpr(1), AffineDimExpr(2))
    assert disabled_pattern.bounds == expected_bounds
    assert disabled_pattern.pattern.results == expected_results
    assert isinstance(disabled_pattern, AccessPattern)

    # test 2: disable 1 dims
    disabled_pattern = access_pattern.disable_dims(1)
    expected_bounds = (20, 30)
    expected_results = (AffineConstantExpr(0), AffineDimExpr(0), AffineDimExpr(1))
    assert disabled_pattern.bounds == expected_bounds
    assert disabled_pattern.pattern.results == expected_results
    assert isinstance(disabled_pattern, AccessPattern)

    # test 3: disable 2 dims
    disabled_pattern = access_pattern.disable_dims(2)
    expected_bounds = (30,)
    expected_results = (AffineConstantExpr(0), AffineConstantExpr(0), AffineDimExpr(0))
    assert disabled_pattern.bounds == expected_bounds
    assert disabled_pattern.pattern.results == expected_results
    assert isinstance(disabled_pattern, AccessPattern)

    # test 4: disable 3 dims (all)
    disabled_pattern = access_pattern.disable_dims(3)
    expected_bounds = tuple()
    expected_results = (AffineConstantExpr(0), AffineConstantExpr(0), AffineConstantExpr(0))
    assert disabled_pattern.bounds == expected_bounds
    assert disabled_pattern.pattern.results == expected_results
    assert isinstance(disabled_pattern, AccessPattern)


def test_access_pattern_tile_dim():
    pattern = AffineMap(
        num_dims=3,
        num_symbols=0,
        results=(AffineDimExpr(0), AffineDimExpr(1), AffineDimExpr(2))
    )
    bounds = (10, 20, 30)
    access_pattern = AccessPattern(bounds, pattern)
    tiled_pattern = access_pattern.tile_dim(1, 5)
    expected_bounds = (10, 5, 4, 30)
    expected_results = (AffineDimExpr(0), AffineDimExpr(1) * 5 + AffineDimExpr(2), AffineDimExpr(3))
    assert tiled_pattern.bounds == expected_bounds
    assert tiled_pattern.pattern.results == expected_results
    assert isinstance(tiled_pattern, AccessPattern)

def test_template_pattern_creation():
    pattern = AffineMap(
        num_dims=2,
        num_symbols=0,
        results=(AffineDimExpr(0), AffineDimExpr(1))
    )
    bounds = (5, 10)
    template_pattern = TemplatePattern(bounds, pattern)
    assert template_pattern.bounds == bounds
    assert template_pattern.pattern == pattern
    assert template_pattern.num_dims == 2

def test_template_pattern_tile_dim_exception():
    pattern = AffineMap(
        num_dims=2,
        num_symbols=0,
        results=(AffineDimExpr(0), AffineDimExpr(1))
    )
    bounds = (5, 10)
    template_pattern = TemplatePattern(bounds, pattern)
    with pytest.raises(RuntimeError, match="A template should not be tiled"):
        template_pattern.tile_dim(0, 2)

def test_template_pattern_rotate_exception():
    pattern = AffineMap(
        num_dims=2,
        num_symbols=0,
        results=(AffineDimExpr(0), AffineDimExpr(1))
    )
    bounds = (5, 10)
    template_pattern = TemplatePattern(bounds, pattern)
    with pytest.raises(RuntimeError, match="A template should not be rotated"):
        template_pattern.rotate(1)

def test_schedule_pattern_creation():
    pattern = AffineMap(
        num_dims=2,
        num_symbols=0,
        results=(AffineDimExpr(0), AffineDimExpr(1))
    )
    bounds = (15, 25)
    schedule_pattern = SchedulePattern(bounds, pattern)
    assert schedule_pattern.bounds == bounds
    assert schedule_pattern.pattern == pattern
    assert schedule_pattern.num_dims == 2
    assert isinstance(schedule_pattern.bounds, tuple)
    assert all(isinstance(b, int) for b in schedule_pattern.bounds)

def test_schedule_pattern_invalid_bounds():
    pattern = AffineMap(
        num_dims=2,
        num_symbols=0,
        results=(AffineDimExpr(0), AffineDimExpr(1))
    )
    with pytest.raises(ValueError, match="All bounds must be static, strictly positive integers for a schedule"):
        SchedulePattern((10, None), pattern) #pyright: ignore
    with pytest.raises(ValueError, match="All bounds must be static, strictly positive integers for a schedule"):
        SchedulePattern((10, -5), pattern)
