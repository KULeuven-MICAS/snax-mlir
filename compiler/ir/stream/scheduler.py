from compiler.ir.stream import Schedule, Template


def scheduler(template: Template, schedule: Schedule) -> Schedule:
    print("Start ===========================================")


    for i in range(template.num_dims):
        # i = 0: look at the last dimension
        # i = 1: look at the second to last dimension
        template_dim = template.num_dims - i - 1
        schedule_dim = schedule.num_dims - i - 1
        match = False


        print("Template:")
        print(template[0].pattern)
        print(template[1].pattern)
        if len(template) > 2:
            print(template[2].pattern)
        print(template.bounds)
        print("Schedule:")
        print(schedule[0].pattern)
        print(schedule[1].pattern)
        if len(schedule) > 2:
            print(schedule[2].pattern)
        print(schedule.bounds)

        # maximum number of rotations
        for _ in range(schedule_dim + 1):
            # check if there is a match
            template_check = template.disable_dims(template_dim)
            schedule_check = schedule.disable_dims(schedule_dim)

            print("Template check:")
            print(template_check[0].pattern)
            print(template_check[1].pattern)
            if len(template_check) > 2:
                print(template_check[2].pattern)
            print(template_check.bounds)
            print("Schedule check:")
            print(schedule_check[0].pattern)
            print(schedule_check[1].pattern)
            if len(schedule_check) > 2:
                print(schedule_check[2].pattern)
            print(schedule_check.bounds)

            print(template_check.matches(schedule_check))

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
                if schedule_bound % template_bound != 0:
                    breakpoint()
                assert schedule_bound % template_bound == 0
                schedule = schedule.tile_dim(schedule_dim, template_bound)
                # nice, continue
                match = True
                break

        if not match:
            breakpoint()
            raise RuntimeError("failed to match template and schedule")

    print("Final Template:")
    print(template[0].pattern)
    print(template[1].pattern)
    if len(template) > 2:
        print(template[2].pattern)
    print(template.bounds)
    print("Final Schedule:")
    print(schedule[0].pattern)
    print(schedule[1].pattern)
    if len(schedule) > 2:
        print(schedule[2].pattern)
    print(schedule.bounds)

    return schedule
