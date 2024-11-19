from collections.abc import Iterable, Iterator

from compiler.ir.stream import Schedule, Template
from compiler.ir.stream.access_pattern import SchedulePattern
from compiler.util.multiset import Multiset


def scheduler_backtrack(template: Template, schedule: Schedule, pure_output_stationary: bool, dim=1) -> Iterator[Schedule]:
    # print(f'Running Scheduler Backtracking for dim = {dim}')
    # print('Schedule:')
    # print(schedule)
    # print('Template:')
    # print(template)

    if dim - 1 >= schedule.num_dims:
        yield schedule

    N = schedule.num_dims - dim
    K = template.num_dims - dim

    for n in range(N + 1):
        # print('Checking the following schedule:')
        # print(schedule)

        if K > 0:
            schedule_check = schedule.disable_dims(N)
            template_check = template.disable_dims(K)
        else:
            schedule_check = schedule.disable_dims(schedule.num_dims - template.num_dims)
            template_check = template

        # print('Template check:')
        # print(template_check)
        # print('Schedule check:')
        # print(schedule_check)

        if template_check.matches(schedule_check):
            if dim > template.num_dims:
                # programmatic acces, should be fine from now on
                # extra check: constrain to output-stationary
                i = schedule.num_dims - dim
                ok = True

                # make sure to be at least one output stationary
                if dim == template.num_dims + 1:
                    if schedule[-1].depends_on(i):
                        ok = False

                # extra check (1): constrain to pure output stationary
                if pure_output_stationary:
                    if schedule[-1].depends_on(i):
                        # no further reductions can be allowed
                        while i >= 0:
                            if not schedule[-1].depends_on(i):
                                # print('not output stationary!')
                                ok = False
                            i -= 1

                # extra check (2): make sure there is correct memory flexibility
                def generate_one_list(n: int, i: int):
                    return [1 if j == i else 0 for j in range(n)]

                # only keep spatial dims:
                for sp in schedule:
                    res = sp.disable_dims(schedule.num_dims - template.num_dims).pattern.eval(
                        [1] * template.num_dims, ()
                    )
                    # for these dimensions, more only one of the upper loops can have
                    # something not divisible by 8
                    nbs_left = 1
                    for idx in [index for index, value in enumerate(res) if value > 0]:
                        for i in range(schedule.num_dims - template.num_dims):
                            result = sp.pattern.eval(generate_one_list(schedule.num_dims, i), ())[idx]
                            if result % 8 != 0:
                                nbs_left -= 1
                    if nbs_left < 0:
                        # print('not legal for memory!')
                        ok = False

                if ok:
                    pass
                    yield from scheduler_backtrack(template, schedule, pure_output_stationary, dim + 1)

            else:
                # check bounds
                template_bound = template[0].bounds[-dim]
                assert template_bound  # must have bound from now on
                schedule_bound = schedule[0].bounds[-dim]

                if schedule_bound == template_bound:
                    pass
                    # print('perfect match, check behaviour')
                elif schedule_bound < template_bound:
                    pass
                    # print('applying padding...')
                    # apply padding:
                    padded_schedule = schedule.pad_dim(N, template_bound)
                    # otherwise:
                    yield from scheduler_backtrack(template, schedule, pure_output_stationary, dim + 1)
                elif schedule_bound > template_bound:
                    if schedule_bound % template_bound != 0:
                        pass
                        # print('imperfect factorization, no support yet')
                        padded_schedule = schedule.pad_dim(N, schedule_bound + (schedule_bound % template_bound))
                        tiled_schedule = padded_schedule.tile_dim(N, template_bound)
                        # try again with padded schedule, but no increased dim
                        yield from scheduler_backtrack(template, tiled_schedule, pure_output_stationary, dim + 1)
                    else:
                        pass
                        # print('match, will apply tiling')
                        tiled_schedule = schedule.tile_dim(N, template_bound)
                        yield from scheduler_backtrack(template, tiled_schedule, pure_output_stationary, dim + 1)
        else:
            pass
            # print('no match')

        # print('rotating...')
        schedule = schedule.rotate(N + 1)


def scheduler(template: Template, schedule: Schedule, schedule_idx: int = 0, pure_output_stationary: bool = True) -> Schedule:
    # prune away the 1-bounded dimensions:
    schedule = schedule.clear_unused_dims()

    schedules = scheduler_backtrack(template, schedule, True)

    schedules = list(schedules)

    # match at schedule idx
    return schedules[schedule_idx]
