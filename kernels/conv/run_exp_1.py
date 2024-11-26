from math import prod
from pprint import pp
import pandas as pd

from genkernel import ConvSpec
from xdsl.ir.affine import AffineDimExpr, AffineMap

from compiler.ir.autoflow.scheduler import scheduler_backtrack
from compiler.ir.stream.access_pattern import Schedule, SchedulePattern, Template, TemplatePattern

# Experiment 1: get spatial mapping efficiencies for different possible mappings.


def run_exeriment(spec: ConvSpec, layertype, array):
    (
        m,
        n,
    ) = (AffineDimExpr(i) for i in range(2))

    template = [
        AffineMap(2, 0, (m, n)),
        AffineMap(2, 0, (m, n)),
        AffineMap(2, 0, (m, n)),
    ]
    template_bounds = (32, 16)
    template = Template(TemplatePattern(template_bounds, tp) for tp in template)

    if array == '3d':
    # overwrite template for gemmx:
        m, n, k = (AffineDimExpr(i) for i in range(3))
        template = [
            AffineMap(3, 0, (m, k, n)),
            AffineMap(3, 0, (k, n, m)),
            AffineMap(3, 0, (m, n, k)),
        ]
        template_bounds = (8, 8, 8)

        template = Template(TemplatePattern(template_bounds, tp) for tp in template)

    elif array == 'gemm':
        m, n, k = (AffineDimExpr(i) for i in range(3))
        template = [
            AffineMap(3, 0, (m, k)),
            AffineMap(3, 0, (k, n)),
            AffineMap(3, 0, (m, n)),
        ]
        template_bounds = (8, 8, 8)

        template = Template(TemplatePattern(template_bounds, tp) for tp in template)

    if spec.depthwise:
        ox, oy, fx, fy, c = (AffineDimExpr(i) for i in range(5))
        schedule_bounds = (spec.ox, spec.oy, spec.fx, spec.fy, spec.c)
        schedule = [
            AffineMap(5, 0, (c, ox + fx, oy + fy)),
            AffineMap(5, 0, (c, fx, fy)),
            AffineMap(5, 0, (c, ox, oy)),
        ]
    else:
        ox, oy, fx, fy, c, k = (AffineDimExpr(i) for i in range(6))
        schedule_bounds = (spec.ox, spec.oy, spec.fx, spec.fy, spec.c, spec.k)
        schedule = [
            AffineMap(6, 0, (c, ox + fx, oy + fy)),
            AffineMap(6, 0, (c, k, fx, fy)),
            AffineMap(6, 0, (k, ox, oy)),
        ]

    schedule = Schedule(SchedulePattern(schedule_bounds, sp) for sp in schedule)

    resulting_schedules = list(
        scheduler_backtrack(template, schedule, early_spatial_exit=True, apply_memory_check=False)
    )

    breakpoint()

    utilization = [
        {"schedule_idx": i, "layer_type": layertype, "array": array, "schedule": str(schedule[0]), "utilizaiton": prod(schedule[0].bounds[-template.num_dims:]) / prod(template_bounds)}
        for i, schedule in enumerate(resulting_schedules)
    ]

    return utilization



if __name__ == "__main__":
    run_exeriment(ConvSpec(1, 32, 32, 7, 7, 192, 192, depthwise=True), "depthwise", "gemm")
    result = []
    for array in ('2d', '3d', 'gemm'):
        result += run_exeriment(ConvSpec(1, 256, 256, 4, 4, 3, 96), "early", array)
        result += run_exeriment(ConvSpec(1, 32, 32, 7, 7, 192, 192, depthwise=True), "depthwise", array)
        result += run_exeriment(ConvSpec(1, 32, 32, 1, 1, 192, 786, ), "pointwise", array)
        result += run_exeriment(ConvSpec(1, 7, 7, 3, 3, 512, 512), "late", array)
    df = pd.DataFrame(result)
    df.to_csv("experiment_1_results.csv", index=False)

