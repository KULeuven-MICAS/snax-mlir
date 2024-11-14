from collections.abc import Iterable, Iterator

from compiler.ir.stream import Schedule, Template


def scheduler_backtrack(template: Template, schedule: Schedule, dim = 1) -> Iterator[Schedule]:

    # print(f'Running Scheduler Backtracking for dim = {dim}')
    # print('Schedule:')
    # print(schedule)
    # print('Template:')
    # print(template)

    if dim >= schedule.num_dims:
        yield schedule

    N = schedule.num_dims - dim
    K = template.num_dims - dim

    for n in range(N):

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
                if schedule[-1].depends_on(i):
                    # no further reductions can be allowed
                    while i >= 0:
                        if not schedule[-1].depends_on(i):
                            ok = False
                        i -= 1
                if ok:
                    yield from scheduler_backtrack(template, schedule, dim + 1)

            else:
                # check bounds
                template_bound = template[0].bounds[-dim]
                assert template_bound # must have bound from now on
                schedule_bound = schedule[0].bounds[-dim]

                if schedule_bound == template_bound:
                    pass
                    # print('perfect match, check behaviour')
                elif schedule_bound < template_bound:
                    pass
                    # print('underutilized array, no support yet')
                    # otherwise:
                    # yield from scheduler_backtrack(template, schedule, dim + 1)
                elif schedule_bound > template_bound:
                    if schedule_bound % template_bound != 0:
                        pass
                        # print('imperfect factorization, no support yet')
                    else:
                        # print('match, will apply tiling')
                        schedule = schedule.tile_dim(N, template_bound)
                        yield from scheduler_backtrack(template, schedule, dim + 1)
        else:
            pass
            # print('no match')

        # print('rotating...')
        schedule = schedule.rotate(N + 1)


def scheduler(template: Template, schedule: Schedule) -> Schedule:

    schedules = scheduler_backtrack(template, schedule)

    schedules = list(schedules)

    # return first match
    return schedules[0]
