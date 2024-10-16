import pytest
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap

from compiler.ir.stream import (
    AccessPattern,
    Schedule,
    SchedulePattern,
    Template,
    TemplatePattern,
)


# Pytest tests
def test_access_pattern_creation():
    pattern = AffineMap(
        num_dims=3,
        num_symbols=0,
        results=(AffineDimExpr(0), AffineDimExpr(1), AffineDimExpr(2)),
    )
    bounds = (10, 20, 30)
    access_pattern = AccessPattern(bounds, pattern)
    assert access_pattern.bounds == bounds
    assert access_pattern.pattern == pattern
    assert access_pattern.num_dims == 3


def test_schedule_pattern_rotate():
    pattern = AffineMap(
        num_dims=3,
        num_symbols=0,
        results=(AffineDimExpr(0), AffineDimExpr(1), AffineDimExpr(2)),
    )
    bounds = (10, 20, 30)
    access_pattern = SchedulePattern(bounds, pattern)

    # test 1: 3 dims, rotate 2
    rotated_pattern = access_pattern.rotate(2)
    expected_bounds = (20, 10, 30)
    expected_results = (AffineDimExpr(1), AffineDimExpr(0), AffineDimExpr(2))
    assert rotated_pattern.bounds == expected_bounds
    assert rotated_pattern.pattern.results == expected_results
    assert isinstance(rotated_pattern, AccessPattern)

    # test 2: 3 dims, rotate 3
    rotated_pattern = access_pattern.rotate(3)
    expected_bounds = (20, 30, 10)
    expected_results = (AffineDimExpr(2), AffineDimExpr(0), AffineDimExpr(1))
    assert rotated_pattern.bounds == expected_bounds
    assert rotated_pattern.pattern.results == expected_results
    assert isinstance(rotated_pattern, AccessPattern)

    # test 3: 3 dims, rotate 1
    rotated_pattern = access_pattern.rotate(1)
    expected_bounds = (10, 20, 30)
    expected_results = (AffineDimExpr(0), AffineDimExpr(1), AffineDimExpr(2))
    assert rotated_pattern.bounds == expected_bounds
    assert rotated_pattern.pattern.results == expected_results
    assert isinstance(rotated_pattern, AccessPattern)

    # test 4 dims
    pattern = AffineMap(
        num_dims=4,
        num_symbols=0,
        results=(AffineDimExpr(0), AffineDimExpr(1), AffineDimExpr(2), AffineDimExpr(3)),
    )
    bounds = (10, 20, 30, 40)
    access_pattern = SchedulePattern(bounds, pattern)

    # test 4: 4 dims, rotate 3
    rotated_pattern = access_pattern.rotate(3)
    expected_bounds = (20, 30, 10, 40)
    expected_results = (AffineDimExpr(2), AffineDimExpr(0), AffineDimExpr(1), AffineDimExpr(3))
    assert rotated_pattern.bounds == expected_bounds
    assert rotated_pattern.pattern.results == expected_results
    assert isinstance(rotated_pattern, AccessPattern)




def test_access_pattern_disable_dims():
    pattern = AffineMap(
        num_dims=3,
        num_symbols=0,
        results=(AffineDimExpr(0), AffineDimExpr(1), AffineDimExpr(2)),
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
    expected_results = (
        AffineConstantExpr(0),
        AffineConstantExpr(0),
        AffineConstantExpr(0),
    )
    assert disabled_pattern.bounds == expected_bounds
    assert disabled_pattern.pattern.results == expected_results
    assert isinstance(disabled_pattern, AccessPattern)


def test_schedule_pattern_tile_dim():
    pattern = AffineMap(
        num_dims=3,
        num_symbols=0,
        results=(AffineDimExpr(0), AffineDimExpr(1), AffineDimExpr(2)),
    )
    bounds = (10, 20, 30)
    access_pattern = SchedulePattern(bounds, pattern)
    tiled_pattern = access_pattern.tile_dim(1, 5)
    expected_bounds = (10, 4, 5, 30)
    expected_results = (
        AffineDimExpr(0),
        AffineDimExpr(1) * 5 + AffineDimExpr(2),
        AffineDimExpr(3),
    )
    assert tiled_pattern.bounds == expected_bounds
    assert tiled_pattern.pattern.results == expected_results
    assert isinstance(tiled_pattern, AccessPattern)


def test_template_pattern_creation():
    pattern = AffineMap(
        num_dims=2, num_symbols=0, results=(AffineDimExpr(0), AffineDimExpr(1))
    )
    bounds = (5, 10)
    template_pattern = TemplatePattern(bounds, pattern)
    assert template_pattern.bounds == bounds
    assert template_pattern.pattern == pattern
    assert template_pattern.num_dims == 2


def test_schedule_pattern_creation():
    pattern = AffineMap(
        num_dims=2, num_symbols=0, results=(AffineDimExpr(0), AffineDimExpr(1))
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
        num_dims=2, num_symbols=0, results=(AffineDimExpr(0), AffineDimExpr(1))
    )
    with pytest.raises(
        ValueError,
        match="All bounds must be static, strictly positive integers for a schedule",
    ):
        SchedulePattern((10, None), pattern)  # pyright: ignore
    with pytest.raises(
        ValueError,
        match="All bounds must be static, strictly positive integers for a schedule",
    ):
        SchedulePattern((10, -5), pattern)


def test_template_pattern_matches():
    pattern = AffineMap(
        num_dims=2, num_symbols=0, results=(AffineDimExpr(0), AffineDimExpr(1))
    )
    bounds = (10, 20)
    tp = TemplatePattern(bounds, pattern)
    sp_matching = SchedulePattern(bounds, pattern)
    sp_non_matching_pattern = SchedulePattern(
        bounds,
        AffineMap(
            num_dims=2, num_symbols=0, results=(AffineDimExpr(1), AffineDimExpr(0))
        ),
    )
    sp_non_matching_bounds = SchedulePattern((5, 15), pattern)

    assert tp.matches(sp_matching) is True
    assert tp.matches(sp_non_matching_pattern) is False
    assert (
        tp.matches(sp_non_matching_bounds) is True
    )  # Bounds are not checked in matches


def test_schedule_rotate():
    pattern1 = AffineMap(
        num_dims=2, num_symbols=0, results=(AffineDimExpr(0), AffineDimExpr(1))
    )
    pattern2 = AffineMap(
        num_dims=2, num_symbols=0, results=(AffineDimExpr(1), AffineDimExpr(0))
    )
    sp1 = SchedulePattern((10, 20), pattern1)
    sp2 = SchedulePattern((30, 40), pattern2)
    schedule = Schedule([sp1, sp2])
    rotated_schedule = schedule.rotate(1)
    assert isinstance(rotated_schedule, Schedule)
    assert rotated_schedule[0].bounds == sp1.rotate(1).bounds
    assert rotated_schedule[1].bounds == sp2.rotate(1).bounds


def test_schedule_disable_dims():
    pattern1 = AffineMap(
        num_dims=3,
        num_symbols=0,
        results=(AffineDimExpr(0), AffineDimExpr(1), AffineDimExpr(2)),
    )
    sp1 = SchedulePattern((10, 20, 30), pattern1)
    schedule = Schedule([sp1])
    disabled_schedule = schedule.disable_dims(2)
    assert isinstance(disabled_schedule, Schedule)
    assert disabled_schedule[0].bounds == sp1.disable_dims(2).bounds


def test_schedule_tile_dim():
    pattern1 = AffineMap(
        num_dims=2, num_symbols=0, results=(AffineDimExpr(0), AffineDimExpr(1))
    )
    sp1 = SchedulePattern((100, 200), pattern1)
    schedule = Schedule([sp1])
    tiled_schedule = schedule.tile_dim(0, 10)
    assert isinstance(tiled_schedule, Schedule)
    expected_bounds = sp1.tile_dim(0, 10).bounds
    assert tiled_schedule[0].bounds == expected_bounds


def test_template_disable_dims():
    pattern1 = AffineMap(
        num_dims=3,
        num_symbols=0,
        results=(AffineDimExpr(0), AffineDimExpr(1), AffineDimExpr(2)),
    )
    tp1 = TemplatePattern((10, 20, 30), pattern1)
    template = Template([tp1])
    disabled_template = template.disable_dims(1)
    assert isinstance(disabled_template, Template)
    assert disabled_template[0].bounds == tp1.disable_dims(1).bounds


def test_template_matches_schedule():
    pattern1 = AffineMap(
        num_dims=2, num_symbols=0, results=(AffineDimExpr(0), AffineDimExpr(1))
    )
    tp1 = TemplatePattern((10, 20), pattern1)
    tp2 = TemplatePattern((30, 40), pattern1)
    template = Template([tp1, tp2])

    sp1 = SchedulePattern((10, 20), pattern1)
    sp2 = SchedulePattern((30, 40), pattern1)
    schedule_matching = Schedule([sp1, sp2])

    sp3 = SchedulePattern(
        (10, 20),
        AffineMap(
            num_dims=2, num_symbols=0, results=(AffineDimExpr(1), AffineDimExpr(0))
        ),
    )
    schedule_non_matching = Schedule([sp1, sp3])

    assert template.matches(schedule_matching) is True
    assert template.matches(schedule_non_matching) is False


def test_template_matches_schedule_length_mismatch():
    pattern1 = AffineMap(num_dims=1, num_symbols=0, results=(AffineDimExpr(0),))
    tp1 = TemplatePattern((10,), pattern1)
    template = Template([tp1])

    sp1 = SchedulePattern((10,), pattern1)
    sp2 = SchedulePattern((20,), pattern1)
    schedule = Schedule([sp1, sp2])

    assert template.matches(schedule) is False
