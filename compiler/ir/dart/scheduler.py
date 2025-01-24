from collections.abc import Generator

from compiler.ir.dart.access_pattern import Schedule, Template


def scheduler_backtrack(
    template: Template,
    schedule: Schedule,
    inner_dims: int = 1,
) -> Generator[Schedule]:
    """
    Backtracking method to find all possible mappings of the schedule on the template

    `template` (Template): the accelerator template
    `schedule` (Schedule): the partially scheduled operation
    `inner_dims` (int): current number of innermost dimensions being handled
    `pure_output_stationary` (bool):
    """

    """
    Explanation of dimensions:

        In case we are handling 6 dimensions and `dim` = 3:

        There are 3 innermost dimensions that are being checked with the template.
        The other outermost dimensions are not considered.

        The current dimension under foces (d3) is the most outermost dim of the innermost dims.
        When we apply a tiling, this will happen to this dimension d3.
        When we apply a rotation, we will rotate outermost dims + focused dim (d0 - d4)

                               +---- `dim` innermost dims
                               |
                    -----------+
         d0, d1, d2, d3, d4, d5
                    --+
        -----------+  +-------------- focused dim
                   |
                   +----------------- `outermost` dims
    """

    # exit condition for the algorithm: if all dimensions are considered
    if inner_dims > schedule.num_dims:
        yield schedule

    # This for loop rotates the outermost + focused dims loops in all ways.
    # There are thus `schedule.num_dims - inner_dims + 1` different rotations possible.
    for _ in range(schedule.num_dims - inner_dims + 1):
        # apply rotation:
        schedule = schedule.rotate(schedule.num_dims - inner_dims + 1)

        # use innermost dimensions for template check
        schedule_check = schedule.inner_dims(inner_dims)
        template_check = template.inner_dims(inner_dims)

        # check 1: check for valid transformation
        if not template_check.matches(schedule_check):
            # not possible, consider next option
            continue

        # checks passed, we have a candidate schedule now
        candidate_schedule = schedule

        # check 2: check for valid iteration bounds
        template_bound = (
            template[0].bounds[-inner_dims] if inner_dims <= template.num_dims else None
        )
        schedule_bound = candidate_schedule[0].bounds[-inner_dims]

        if template_bound:
            if schedule_bound < template_bound:
                # TODO: underutilized array, apply padding
                continue
            elif schedule_bound % template_bound != 0:
                # TODO: imperfect factorization
                continue
            else:  # >=
                # tile schedule
                candidate_schedule = candidate_schedule.tile_dim(
                    schedule.num_dims - inner_dims, template_bound
                )

        # continue with candidate schedule, with an extra inner dim:
        yield from scheduler_backtrack(template, candidate_schedule, inner_dims + 1)


def scheduler(template: Template, schedule: Schedule) -> Schedule:
    # for now just return the first result of the backtracking
    result = next(scheduler_backtrack(template, schedule))
    return result
