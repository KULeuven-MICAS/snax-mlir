from compiler.ir.stream import Schedule, Template


def scheduler(template: Template, schedule: Schedule) -> Schedule:
    for i in range(template.num_dims):
        # i = 0: look at the last dimension
        # i = 1: look at the second to last dimension
        template_dim = template.num_dims - i - 1
        schedule_dim = schedule.num_dims - i - 1
        match = False

        # maximum number of rotations
        for _ in range(schedule_dim + 1):
            # check if there is a match
            template_check = template.disable_dims(template_dim)
            schedule_check = schedule.disable_dims(schedule_dim)

            if template_check.matches(schedule_check):
                match = True
                break

            # else rotate the for loops
            schedule = schedule.rotate(schedule_dim + 1)

        if not match:
            raise RuntimeError("failed to match template and schedule")

        # now, check bounds and design potential transformation map
        if not (template_bound := template[0].bounds[template_dim]):
            # nothing to worry about, continue to next dim
            continue

        schedule_bound = schedule[0].bounds[schedule_dim]

        if schedule_bound < template_bound:
            # need to apply padding
            raise NotImplementedError("padding not supported")
        elif schedule_bound > template_bound:
            # need to split up the schedule
            assert schedule_bound % template_bound == 0
            schedule = schedule.tile_dim(schedule_dim, template_bound)

    return schedule
