from collections.abc import Callable, Iterator, Sequence
from math import ceil

import numpy as np

from snaxc.ir.dart.access_pattern import Schedule, Template


def scheduler_backtrack(
    template: Template,
    schedule: Schedule,
    inner_dims: int = 1,
    extra_checks: Sequence[Callable[[Template, Schedule], bool]] = [],
) -> Iterator[Schedule]:
    """
    Backtracking method to find all possible mappings of the schedule on the template

    `template` (Template): the accelerator template
    `schedule` (Schedule): the partially scheduled operation
    `inner_dims` (int): current number of innermost dimensions being handled
    """

    """
    Explanation of dimensions:

        In case we are handling 6 dimensions and `dim` = 3:

        There are 3 innermost dimensions that are being checked with the template.
        The other outermost dimensions are not considered.

        The current dimension under consideration (d3) is the most outermost dim of the innermost dims.
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

        # check 2: apply extra checks
        if not all(check(template_check, schedule_check) for check in extra_checks):
            # not a valid schedule, consider next option
            continue

        # checks passed, we have a candidate schedule now
        candidate_schedule = schedule

        # check 3: check for valid iteration bounds
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
        yield from scheduler_backtrack(
            template, candidate_schedule, inner_dims + 1, extra_checks
        )


def is_pure_output_stationary(template: Template, schedule: Schedule):
    """
    Checks whether a schedule, outside of the template, is fully output
    stationary. This is determined by making sure all parallel dimensions
    precede the reduction dimensions in the output operand (last operand).
    """
    # fetch the pattern of the last operand
    output_schedule = schedule[-1].pattern.A
    # do not consider template dims
    output_schedule = output_schedule[:, : -template.num_dims]

    # check whether there are any non-zero elements in every column
    # create iteration_types list with False for reduction, True for parallel
    iteration_types: list[bool] = list(
        map(lambda x: bool(x), np.any(output_schedule != 0, axis=0).tolist())
    )
    # the first zero should come after the last 1 for output stationary

    # if only reduction, or only parallel, pure otuput stationary is guaranteed
    if not (True in iteration_types and False in iteration_types):
        return True

    first_reduction_idx = iteration_types.index(False)
    last_parallel_idx = len(iteration_types) - 1 - iteration_types[::-1].index(True)

    # last parallel index should come before first reduction idx for pure output stationarity
    return first_reduction_idx > last_parallel_idx


def is_memory_flexible_enough(
    template: Template, schedule: Schedule, element_sizes: Sequence[int]
):
    """
    Checks whether the TCDM flexibility is sufficient to actually execute
    the schedule.

    There must be one spatial stride of 1 that doesn't need more fine-grained
    temporal access within one bank, such that that dimension can be packed together.
    """
    TCDM_BANK_WIDTH = 8
    # We can only apply this check if there are temporal dimensions to investigate
    # their access granularity:
    if not schedule.num_dims > template.num_dims:
        return True
    for s, size in zip(schedule, element_sizes):
        # is there temporary fine-grained access for this dimension?
        temporal = (
            s.pattern.A[:, 0 : -template.num_dims] % ceil(TCDM_BANK_WIDTH / size)
        ).any(axis=1)
        # is the dimension spatially unrolled?
        spatial = (s.pattern.A[:, -template.num_dims :] == 1).any(axis=1)
        if (False, True) not in zip(temporal, spatial):
            return False
    return True


def scheduler(
    template: Template,
    schedule: Schedule,
    extra_checks: Sequence[Callable[[Template, Schedule], bool]] = [
        # defaulting to pure output stationary schedules for now
        is_pure_output_stationary,
    ],
    schedule_idx: int | None = None,
) -> Schedule:
    # for now just return the first result of the backtracking
    if schedule_idx is not None:
        all = list(scheduler_backtrack(template, schedule, extra_checks=extra_checks))
        return all[schedule_idx]
    result = next(scheduler_backtrack(template, schedule, extra_checks=extra_checks))
    return result
