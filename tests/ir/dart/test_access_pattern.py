import numpy as np
import pytest
from xdsl.ir.affine import AffineDimExpr, AffineMap

from compiler.ir.dart.access_pattern import (
    AccessPattern,
    Schedule,
    SchedulePattern,
    Template,
    TemplatePattern,
)
from compiler.ir.dart.affine_transform import AffineTransform


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
    assert access_pattern.pattern == AffineTransform.from_affine_map(pattern)
    assert access_pattern.num_dims == 3


def test_access_pattern_disable_dims():
    pattern = AffineTransform(
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), b=np.array([1, 2, 3])
    )
    bounds = (10, 20, 30)
    access_pattern = AccessPattern(bounds, pattern)

    # test 1: disable 0 dims (none)
    disabled_pattern = access_pattern.disable_dims(0)
    assert disabled_pattern.bounds == bounds
    assert disabled_pattern.pattern == pattern
    assert isinstance(disabled_pattern, AccessPattern)

    # test 2: disable 1 dims
    disabled_pattern = access_pattern.disable_dims(1)
    expected_bounds = (20, 30)
    expected_results = np.array([[0, 0], [1, 0], [0, 1]])
    assert disabled_pattern.bounds == expected_bounds
    assert (disabled_pattern.pattern.A == expected_results).all()
    assert (disabled_pattern.pattern.b == pattern.b).all()
    assert isinstance(disabled_pattern, AccessPattern)

    # test 3: disable 2 dims
    disabled_pattern = access_pattern.disable_dims(2)
    expected_bounds = (30,)
    expected_results = np.array([[0], [0], [1]])
    assert disabled_pattern.bounds == expected_bounds
    assert (disabled_pattern.pattern.A == expected_results).all()
    assert (disabled_pattern.pattern.b == pattern.b).all()
    assert isinstance(disabled_pattern, AccessPattern)

    # test 4: disable 3 dims (all)
    disabled_pattern = access_pattern.disable_dims(3)
    expected_bounds: tuple[int, ...] = tuple()
    expected_results = []
    assert disabled_pattern.bounds == expected_bounds
    assert (disabled_pattern.pattern.A == expected_results).all()
    assert (disabled_pattern.pattern.b == pattern.b).all()
    assert isinstance(disabled_pattern, AccessPattern)


def test_access_pattern_inner_dims():
    pattern = AffineTransform(
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), b=np.array([1, 2, 3])
    )
    bounds = (10, 20, 30)
    access_pattern = AccessPattern(bounds, pattern)

    # test 1: keep all dims
    inner_pattern = access_pattern.inner_dims(3)
    assert inner_pattern.bounds == bounds
    assert inner_pattern.pattern == pattern
    assert isinstance(inner_pattern, AccessPattern)

    # test 2: get 2 dims
    inner_pattern = access_pattern.inner_dims(2)
    expected_bounds = (20, 30)
    expected_results = np.array([[0, 0], [1, 0], [0, 1]])
    assert inner_pattern.bounds == expected_bounds
    assert (inner_pattern.pattern.A == expected_results).all()
    assert (inner_pattern.pattern.b == pattern.b).all()
    assert isinstance(inner_pattern, AccessPattern)

    # test 3: get 1 dim
    inner_pattern = access_pattern.inner_dims(1)
    expected_bounds = (30,)
    expected_results = np.array([[0], [0], [1]])
    assert inner_pattern.bounds == expected_bounds
    assert (inner_pattern.pattern.A == expected_results).all()
    assert (inner_pattern.pattern.b == pattern.b).all()
    assert isinstance(inner_pattern, AccessPattern)

    # request 0 inner dims (invalid)
    with pytest.raises(ValueError):
        access_pattern.inner_dims(0)

    # requesting more inner dims than available should
    # just return the original pattern
    inner_pattern = access_pattern.inner_dims(4)
    assert inner_pattern.bounds == bounds
    assert inner_pattern.pattern == pattern
    assert isinstance(inner_pattern, AccessPattern)


def test_schedule_pattern_creation():
    pattern = AffineTransform(np.array([[1, 0], [0, 1]]), np.array([0, 0]))
    bounds = (15, 25)
    schedule_pattern = SchedulePattern(bounds, pattern)
    assert schedule_pattern.bounds == bounds
    assert schedule_pattern.pattern == pattern
    assert schedule_pattern.num_dims == 2
    assert isinstance(schedule_pattern.bounds, tuple)
    assert all(isinstance(b, int) for b in schedule_pattern.bounds)


def test_schedule_pattern_invalid_bounds():
    pattern = AffineTransform(np.array([[1, 0], [0, 1]]), np.array([0, 0]))
    with pytest.raises(
        ValueError,
        match="All bounds must be static, strictly positive integers for a schedule",
    ):
        SchedulePattern((10, -5), pattern)


def test_schedule_pattern_rotate():
    pattern = AffineTransform(
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0, 0, 0])
    )
    bounds = (10, 20, 30)
    access_pattern = SchedulePattern(bounds, pattern)

    # test 1: 3 dims, rotate 2
    rotated_pattern = access_pattern.rotate(2)
    expected_bounds = (20, 10, 30)
    expected_results = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert rotated_pattern.bounds == expected_bounds
    assert (rotated_pattern.pattern.A == expected_results).all()
    assert (rotated_pattern.pattern.b == pattern.b).all()
    assert isinstance(rotated_pattern, AccessPattern)

    # test 2: 3 dims, rotate 3
    rotated_pattern = access_pattern.rotate(3)
    expected_bounds = (20, 30, 10)
    expected_results = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    assert rotated_pattern.bounds == expected_bounds
    assert (rotated_pattern.pattern.A == expected_results).all()
    assert (rotated_pattern.pattern.b == pattern.b).all()
    assert isinstance(rotated_pattern, AccessPattern)

    # test 3: 3 dims, rotate 1
    rotated_pattern = access_pattern.rotate(1)
    expected_bounds = (10, 20, 30)
    assert rotated_pattern.bounds == expected_bounds
    assert (rotated_pattern.pattern.A == pattern.A).all()
    assert (rotated_pattern.pattern.b == pattern.b).all()
    assert isinstance(rotated_pattern, AccessPattern)

    # test 4 dims
    pattern = AffineTransform(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        np.array([0, 0, 0, 0]),
    )
    bounds = (10, 20, 30, 40)
    access_pattern = SchedulePattern(bounds, pattern)

    # test 4: 4 dims, rotate 3
    rotated_pattern = access_pattern.rotate(3)
    expected_bounds = (20, 30, 10, 40)
    expected_results = np.array(
        [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )
    assert rotated_pattern.bounds == expected_bounds
    assert (rotated_pattern.pattern.A == expected_results).all()
    assert (rotated_pattern.pattern.b == pattern.b).all()
    assert isinstance(rotated_pattern, AccessPattern)


def test_schedule_pattern_add_dim():
    pattern = AffineTransform(
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0, 0, 0])
    )
    bounds = (10, 20, 30)
    access_pattern = SchedulePattern(bounds, pattern)
    pattern_new_dim = access_pattern.add_dim()
    expected_bounds = (1, 10, 20, 30)
    expected_results = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert pattern_new_dim.bounds == expected_bounds
    assert (pattern_new_dim.pattern.A == expected_results).all()
    assert isinstance(pattern_new_dim, SchedulePattern)


def test_schedule_pattern_tile_dim():
    pattern = AffineTransform(
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0, 0, 0])
    )
    bounds = (10, 20, 30)
    access_pattern = SchedulePattern(bounds, pattern)
    tiled_pattern = access_pattern.tile_dim(1, 5)
    expected_bounds = (10, 4, 5, 30)
    expected_results = np.array([[1, 0, 0, 0], [0, 5, 1, 0], [0, 0, 0, 1]])
    assert tiled_pattern.bounds == expected_bounds
    assert (tiled_pattern.pattern.A == expected_results).all()
    assert isinstance(tiled_pattern, SchedulePattern)


def test_template_pattern_creation():
    pattern = AffineTransform(np.array([[1, 0], [0, 1], [0, 1]]), np.array([0, 0, 0]))
    bounds = (5, 10)
    template_pattern = TemplatePattern(bounds, pattern)
    assert template_pattern.bounds == bounds
    assert template_pattern.pattern == pattern
    assert template_pattern.num_dims == 2


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


def test_schedule_clear_unused_dims():
    pattern1 = AffineTransform(np.array([[1, 0, 0], [0, 1, 0]]), np.array([0, 0]))
    sp1 = SchedulePattern((1, 10, 1), pattern1)
    schedule = Schedule([sp1])
    cleared_schedule = schedule.clear_unused_dims()
    assert isinstance(cleared_schedule, Schedule)

    assert cleared_schedule[0].bounds == ((10,))
    expected_results = np.array([[0], [1]])
    assert (cleared_schedule[0].pattern.A == expected_results).all()


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
