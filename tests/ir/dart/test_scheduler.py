from xdsl.ir.affine import AffineMap

from snaxc.ir.dart.access_pattern import (
    Schedule,
    SchedulePattern,
    Template,
    TemplatePattern,
)
from snaxc.ir.dart.scheduler import (
    is_memory_flexible_enough,
    is_pure_output_stationary,
    scheduler,
    scheduler_backtrack,
)


def test_matching_1o():
    # test matching template and schedule for 1 operand

    pattern = AffineMap.from_callable(lambda i, j, k: (i + j + k,))
    template = Template((TemplatePattern(bounds=(4, 4, 4), pattern=pattern),))
    schedule = Schedule((SchedulePattern(bounds=(4, 4, 4), pattern=pattern),))

    resulting_schedule = scheduler(template, schedule, extra_checks=[])

    assert schedule == resulting_schedule.clear_unused_dims()


def test_matching_2o():
    # test matching template and schedule for 2 operands

    pattern = AffineMap.from_callable(lambda i, j, k: (i + j + k,))
    template = Template((TemplatePattern(bounds=(4, 4, 4), pattern=pattern),) * 2)
    schedule = Schedule((SchedulePattern(bounds=(4, 4, 4), pattern=pattern),) * 2)

    resulting_schedule = scheduler(template, schedule, extra_checks=[])

    assert schedule == resulting_schedule.clear_unused_dims()


def test_tiling_1o1_1d():
    # test tiling 1 dimension for 1 operand with 1 dimension

    pattern_template = AffineMap.from_callable(lambda m, n: (2 * m + n,))
    pattern_schedule = AffineMap.from_callable(lambda n: (n,))
    pattern_expected = AffineMap.from_callable(lambda m, n: (2 * m + n,))

    template = Template((TemplatePattern(bounds=(None, 2), pattern=pattern_template),))
    schedule = Schedule((SchedulePattern(bounds=(4,), pattern=pattern_schedule),))
    expected = Schedule((SchedulePattern(bounds=(2, 2), pattern=pattern_expected),))

    result = scheduler(template, schedule, extra_checks=[])

    assert result == expected


def test_tiling_1o1_2d():
    # test tiling 2 dimensions for 1 operand

    pattern_template = AffineMap.from_callable(lambda a, b, c, d: (2 * a + b + 2 * c + d,))
    pattern_schedule = AffineMap.from_callable(lambda b, d: (b + d,))
    pattern_expected = AffineMap.from_callable(lambda a, b, c, d: (2 * a + b + 2 * c + d,))

    template = Template((TemplatePattern(bounds=(None, 2, None, 2), pattern=pattern_template),))
    schedule = Schedule((SchedulePattern(bounds=(4, 4), pattern=pattern_schedule),))
    expected = Schedule((SchedulePattern(bounds=(2, 2, 2, 2), pattern=pattern_expected),))

    result = scheduler(template, schedule, extra_checks=[])

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

    template = Template(TemplatePattern((None, 2, None, 2), pattern) for pattern in pattern_template)
    schedule = Schedule(SchedulePattern((4, 4), pattern) for pattern in pattern_schedule)
    expected = Schedule(SchedulePattern((2, 2, 2, 2), pattern) for pattern in pattern_expected)

    result = scheduler(template, schedule, extra_checks=[])

    assert result == expected


def test_tiling_1o2_1d():
    # test tiling 1 dimension for 2 operands with 1 dimensions

    pattern_template = AffineMap.from_callable(lambda a, b, c, d: (2 * a + b, 2 * c + d))
    pattern_schedule = AffineMap.from_callable(lambda b, d: (b, d))
    pattern_expected = AffineMap.from_callable(lambda a, b, c, d: (2 * a + b, 2 * c + d))

    template = Template((TemplatePattern(bounds=(None, 2, None, 2), pattern=pattern_template),))
    schedule = Schedule((SchedulePattern(bounds=(4, 4), pattern=pattern_schedule),))
    expected = Schedule((SchedulePattern(bounds=(2, 2, 2, 2), pattern=pattern_expected),))

    results = list(scheduler_backtrack(template, schedule, extra_checks=[]))

    assert expected in results


def test_tiling_1o_1d2():
    # test tiling 1 dimension twice
    pattern_template = AffineMap.from_callable(lambda a, b, c: (4 * a + 2 * b + c,))
    pattern_schedule = AffineMap.from_callable(lambda c: (c,))
    pattern_expected = AffineMap.from_callable(lambda a, b, c: (4 * a + 2 * b + c,))

    template = Template((TemplatePattern(bounds=(None, 2, 2), pattern=pattern_template),))
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

    # test if the schedule idx returns the correct index
    assert scheduler(template, schedule, extra_checks=[], schedule_idx=3) == result[3]


def test_pure_output_stationary_check():
    template_pattern = AffineMap.from_callable(lambda e: (e, e, e))

    schedule_checks: list[tuple[AffineMap, bool]] = [
        # only template dim: is valid
        (AffineMap.from_callable(lambda e: (e, e, e)), True),
        # only parallel dims: is valid
        (AffineMap.from_callable(lambda c, d, e: (c + e, d + e, e)), True),
        # only reduction dims: is valid
        (AffineMap.from_callable(lambda _c, _d, e: (e, e, e)), True),
        # parallel dim before reduction dim: valid
        (AffineMap.from_callable(lambda c, _, e: (e, e, 2 * c + e)), True),
        # reduction dim before parallel dim: invalid
        (AffineMap.from_callable(lambda _, d, e: (e, e, 2 * d + e)), False),
        # some more complex mixtures of parallel dim / reduction dim
        (AffineMap.from_callable(lambda a, b, _c, _d, e: (e, b + e, 2 * a + e)), True),
        (AffineMap.from_callable(lambda _a, _b, c, d, e: (e, c + e, 2 * d + e)), False),
        (AffineMap.from_callable(lambda a, _b, c, _d, e: (e, a + e, 2 * c + e)), False),
        (AffineMap.from_callable(lambda _a, b, _c, d, e: (e, b + e, 2 * d + e)), False),
    ]

    template = Template((TemplatePattern([1] * template_pattern.num_dims, template_pattern),))

    for schedule_pattern, expected_result in schedule_checks:
        schedule = Schedule((SchedulePattern([1] * schedule_pattern.num_dims, schedule_pattern),))
        assert is_pure_output_stationary(template, schedule) is expected_result


def test_pure_output_stationary_scheduler():
    template_pattern = AffineMap.from_callable(lambda y: (y,))
    template = Template([TemplatePattern([4], template_pattern)])

    schedule_pattern = AffineMap.from_callable(lambda x, y: (y,))
    schedule = Schedule([SchedulePattern([8, 8], schedule_pattern)])

    # the expected output stationary schedule
    output_stationary_pattern = AffineMap.from_callable(lambda y0, x, y1: (4 * y0 + y1,))
    schedule_output_stationary = Schedule([SchedulePattern([2, 8, 4], output_stationary_pattern)])

    # if we run the scheduler without constraints, there are 2 valid schedules:
    result = list(scheduler_backtrack(template, schedule, extra_checks=[]))
    assert len(result) == 2
    # one of which is the output stationary one:
    assert schedule_output_stationary in result

    # if we run the scheduler with the pure output stationary constraint, there is 1:
    result = list(scheduler_backtrack(template, schedule, extra_checks=[is_pure_output_stationary]))
    assert len(result) == 1
    # that one result being the output stationary one
    assert result[0] == schedule_output_stationary


def test_memory_flexibility_scheduler():
    template_pattern = AffineMap.from_callable(lambda y: (y,))
    template = Template([TemplatePattern([4], template_pattern)])

    schedule_pattern = AffineMap.from_callable(lambda x, y: (x + y,))
    schedule = Schedule([SchedulePattern([3, 8], schedule_pattern)])

    results_without = list(scheduler_backtrack(template, schedule, extra_checks=[]))
    results_with = list(
        scheduler_backtrack(
            template,
            schedule,
            extra_checks=[lambda t, s: is_memory_flexible_enough(t, s, [1])],
        )
    )

    assert len(results_without) == 2
    assert len(results_with) == 0
