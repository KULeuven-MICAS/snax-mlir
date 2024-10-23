from compiler.ir.stream import Schedule, Template


def scheduler(template: Template, schedule: Schedule) -> Schedule:
    for i in range(template.num_dims):
        # i = 0: look at the last dimension
        # i = 1: look at the second to last dimension
        template_dim = template.num_dims - i - 1
        schedule_dim = schedule.num_dims - i - 1
        match = False

        #     print("Template:")
        #     print(template[0].pattern)
        #     print(template[1].pattern)
        #     print(template[2].pattern)
        #     print("Schedule:")
        #     print(schedule[0].pattern)
        #     print(schedule[1].pattern)
        #     print(schedule[2].pattern)

        # maximum number of rotations
        for _ in range(schedule_dim + 1):
            # check if there is a match
            template_check = template.disable_dims(template_dim)
            schedule_check = schedule.disable_dims(schedule_dim)

            #         print("Template check:")
            #         print(template_check[0].pattern)
            #         print(template_check[1].pattern)
            #         print(template_check[2].pattern)
            #         print("Schedule check:")
            #         print(schedule_check[0].pattern)
            #         print(schedule_check[1].pattern)
            #         print(schedule_check[2].pattern)

            #         print(template_check.matches(schedule_check))

            # if template_check.matches(schedule_check):
            #     match = True
            #     break

            if not template_check.matches(schedule_check):
                schedule = schedule.rotate(schedule_dim + 1)
                continue

            # # else rotate the for loops
            # schedule = schedule.rotate(schedule_dim + 1)

            # now, check bounds and design potential transformation map
            if not (template_bound := template[0].bounds[template_dim]):
                # nothing to worry about, continue to next dim
                match = True
                break

            schedule_bound = schedule[0].bounds[schedule_dim]

            if schedule_bound < template_bound:
                # need to apply padding, but not supported yet.
                # try and find other option
                schedule = schedule.rotate(schedule_dim + 1)
                continue
                # raise NotImplementedError("padding not supported")
            elif schedule_bound >= template_bound:
                # need to split up the schedule
                assert schedule_bound % template_bound == 0
                schedule = schedule.tile_dim(schedule_dim, template_bound)
                # nice, continue
                match = True
                break

        if not match:
            raise RuntimeError("failed to match template and schedule")

    # print("Final Template:")
    # print(template[0].pattern)
    # print(template[1].pattern)
    # print(template[2].pattern)
    # print("Final Schedule:")
    # print(schedule[0].pattern)
    # print(schedule[1].pattern)
    # print(schedule[2].pattern)

    return schedule
