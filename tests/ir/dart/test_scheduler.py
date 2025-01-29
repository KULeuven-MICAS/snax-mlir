from xdsl.ir.affine import AffineMap

from compiler.ir.dart.access_pattern import (
    Schedule,
    SchedulePattern,
    Template,
    TemplatePattern,
)
from compiler.ir.dart.scheduler import (
    scheduler,
    scheduler_backtrack,
)


def test_matching_1o():
    # test matching template and schedule for 1 operand

    pattern = AffineMap.from_callable(lambda i, j, k: (i, j, k))
    template = Template((TemplatePattern(bounds=(4, 4, 4), pattern=pattern),))
    schedule = Schedule((SchedulePattern(bounds=(4, 4, 4), pattern=pattern),))

    resulting_schedule = scheduler(template, schedule)

    assert schedule == resulting_schedule.clear_unused_dims()


def test_matching_2o():
    # test matching template and schedule for 2 operands

    pattern = AffineMap.from_callable(lambda i, j, k: (i, j, k))
    template = Template((TemplatePattern(bounds=(4, 4, 4), pattern=pattern),) * 2)
    schedule = Schedule((SchedulePattern(bounds=(4, 4, 4), pattern=pattern),) * 2)

    resulting_schedule = scheduler(template, schedule)

    assert schedule == resulting_schedule.clear_unused_dims()


def test_tiling_1o1_1d():
    # test tiling 1 dimension for 1 operand with 1 dimension

    pattern_template = AffineMap.from_callable(lambda m, n: (2 * m + n,))
    pattern_schedule = AffineMap.from_callable(lambda n: (n,))
    pattern_expected = AffineMap.from_callable(lambda m, n: (2 * m + n,))

    template = Template((TemplatePattern(bounds=(None, 2), pattern=pattern_template),))
    schedule = Schedule((SchedulePattern(bounds=(4,), pattern=pattern_schedule),))
    expected = Schedule((SchedulePattern(bounds=(2, 2), pattern=pattern_expected),))

    result = scheduler(template, schedule)

    assert result == expected


def test_tiling_1o1_2d():
    # test tiling 2 dimensions for 1 operand

    pattern_template = AffineMap.from_callable(
        lambda a, b, c, d: (2 * a + b + 2 * c + d,)
    )
    pattern_schedule = AffineMap.from_callable(lambda b, d: (b + d,))
    pattern_expected = AffineMap.from_callable(
        lambda a, b, c, d: (2 * a + b + 2 * c + d,)
    )

    template = Template(
        (TemplatePattern(bounds=(None, 2, None, 2), pattern=pattern_template),)
    )
    schedule = Schedule((SchedulePattern(bounds=(4, 4), pattern=pattern_schedule),))
    expected = Schedule(
        (SchedulePattern(bounds=(2, 2, 2, 2), pattern=pattern_expected),)
    )

    result = scheduler(template, schedule)

    assert result == expected


def test_tiling_2o1_2d():
    # test tiling 2 dimensions for 2 operands with 1 dimensions

    pattern_template = (
        AffineMap.from_callable(lambda a, b, c, d: (2 * a + b,)),
        AffineMap.from_callable(lambda a, b, c, d: (2 * c + d,)),
    )
    pattern_schedule = (
        AffineMap.from_callable(lambda a, b: (a,)),
        AffineMap.from_callable(lambda a, b: (b,)),
    )
    pattern_expected = (
        AffineMap.from_callable(lambda a, b, c, d: (2 * a + b,)),
        AffineMap.from_callable(lambda a, b, c, d: (2 * c + d,)),
    )

    template = Template(
        TemplatePattern((None, 2, None, 2), pattern) for pattern in pattern_template
    )
    schedule = Schedule(
        SchedulePattern((4, 4), pattern) for pattern in pattern_schedule
    )
    expected = Schedule(
        SchedulePattern((2, 2, 2, 2), pattern) for pattern in pattern_expected
    )

    result = scheduler(template, schedule)

    assert result == expected


def test_tiling_1o2_1d():
    # test tiling 1 dimension for 2 operands with 1 dimensions

    pattern_template = AffineMap.from_callable(
        lambda a, b, c, d: (2 * a + b, 2 * c + d)
    )
    pattern_schedule = AffineMap.from_callable(lambda b, d: (b, d))
    pattern_expected = AffineMap.from_callable(
        lambda a, b, c, d: (2 * a + b, 2 * c + d)
    )

    template = Template(
        (TemplatePattern(bounds=(None, 2, None, 2), pattern=pattern_template),)
    )
    schedule = Schedule((SchedulePattern(bounds=(4, 4), pattern=pattern_schedule),))
    expected = Schedule(
        (SchedulePattern(bounds=(2, 2, 2, 2), pattern=pattern_expected),)
    )

    result = scheduler(template, schedule)

    assert result == expected


def test_tiling_1o_1d2():
    # test tiling 1 dimension twice
    pattern_template = AffineMap.from_callable(lambda a, b, c: (4 * a + 2 * b + c,))
    pattern_schedule = AffineMap.from_callable(lambda c: (c,))
    pattern_expected = AffineMap.from_callable(lambda a, b, c: (4 * a + 2 * b + c,))

    template = Template(
        (TemplatePattern(bounds=(None, 2, 2), pattern=pattern_template),)
    )
    schedule = Schedule((SchedulePattern(bounds=(8,), pattern=pattern_schedule),))
    expected = Schedule((SchedulePattern(bounds=(2, 2, 2), pattern=pattern_expected),))

    result = scheduler(template, schedule)

    assert result == expected


def test_multiple_results():
    # test the scheduling result of a problem with multiple results
    pattern_template = AffineMap.from_callable(lambda c: (c,))
    template = Template((TemplatePattern(bounds=(2,), pattern=pattern_template),))
    pattern_schedule = AffineMap.from_callable(lambda a, b, c: (c,))
    schedule = Schedule((SchedulePattern(bounds=(4, 4, 4), pattern=pattern_schedule),))

    # the c dimension will be tiled, so 3 temporal for loops remain.
    # there should be 3*2*1=6 ways to order them, so we excpect 6 results.

    result = list(scheduler_backtrack(template, schedule))
    assert len(result) == 6
