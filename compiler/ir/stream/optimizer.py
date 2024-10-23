from compiler.ir.stream import Schedule, Template


def optimizer(template: Template, schedule: Schedule) -> Schedule:
    # Optimization 1: put all free-moving reduction dims as the most inner loops

    free_reduction_dims = []

    # free-moving dims are dims that appear in schedule but not template
    for i in range(schedule.num_dims - template.num_dims):
        if not schedule[-1].depends_on(i):
            free_reduction_dims.append(i)

    # apply reordering:
    fixed_dims = len([x for x in template[-1].bounds if x is not None])
    new_order = [x for x in range(schedule.num_dims) if x not in free_reduction_dims]
    new_order[-fixed_dims:-fixed_dims] = free_reduction_dims
    schedule = schedule.reorder(new_order)

    # Optimization 2: clear all unused dims of the pattern (= with bound 1)
    schedule = schedule.clear_unused_dims()

    return schedule
