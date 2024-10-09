from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineExpr, AffineMap

from compiler.dialects import stream
from compiler.dialects.kernel import QMacOp
from compiler.util.canonicalize_affine import canonicalize_map


def disable_dims(map: AffineMap, dim: int) -> AffineMap:
    """
    Returns an affine map with the leftmost `dim` dimensions set to 0

    For example:
        (d0, d1, d2) -> d0 + d1 + d2
    For `dim` = 1, will return:
        (d1, d2) -> d1 + d2
    For `dim` = 2, will return:
        (d2) -> d2
    """
    return canonicalize_map(
        map.replace_dims_and_symbols(
            [AffineConstantExpr(0) for _ in range(dim)]
            + [AffineDimExpr(i) for i in range(map.num_dims - dim)],
            [],
            map.num_dims - dim,
            0,
        )
    )


def rotate_dims(map: AffineMap, dim: int) -> AffineMap:
    """
    Returns an affine map with the leftmost `dim` dimensions rotated

    For example:
        (d0, d1, d2) -> 1 * d0 + 2 * d1 + 3 * d2
    For `dim` = 3, will return:
        (d0, d1, d2) -> 3 * d0 + 1 * d1 + 2 * d2
    For `dim` = 2, will return:
        (d0, d1, d2) -> 2 * d0 + 1 * d1 + 3 * d2
    """
    new_dims = [AffineDimExpr(i) for i in range(dim)]
    # rotate dims by popping first and appending
    new_dims.append(new_dims.pop(0))
    # keep remaining dims
    new_dims = new_dims + [AffineDimExpr(i) for i in range(dim, map.num_dims)]
    return canonicalize_map(
        map.replace_dims_and_symbols(new_dims, [], len(new_dims), 0)
    )


def rotate_bounds(bounds: list[int | None], dim: int) -> list[int | None]:
    """
    Returns the bounds after rotating dims, as in rotate_dims
    """
    bounds = bounds.copy()
    bounds.insert(0, bounds.pop(dim - 1))
    return bounds


def tile_dim(map: AffineMap, dim: int, template_bound: int) -> AffineMap:
    """
    Returns the bounds and a new affine map with the `dim` dimension split up into two
    This translates to creating two for loops with adjusted bounds from one for loop


    For example:
        (d0, d1, d2) -> d0 + d1 + d2
    For `dim` = 1, `template_bound` = 2:
        (d0, d1, d2, d3) -> d0 + 2 * d1 + d2 + d3
    """
    # 1 extra dimension
    # create result map (d0, d1, ... dn)
    new_results: list[AffineExpr] = [AffineDimExpr(i) for i in range(map.num_dims + 1)]
    # pop the result at dim
    dim_sum = new_results.pop(dim)
    # add it to dim multiplied by original bound // max_bound
    new_results[dim] = new_results[dim] + dim_sum * template_bound
    transform_map = AffineMap(map.num_dims + 1, 0, tuple(new_results))

    result = canonicalize_map(map.compose(transform_map))

    return result


def tile_bounds(
    bounds: list[int | None], dim: int, template_bound: int
) -> list[int | None]:
    """
    Returns the bounds after applying `tile_dim` in similar fashion.

    For example:
        [2, 8, 2]
    For `dim` = 1, `template_bound` = 2:
        [2, 4, 2, 2]
    """
    bounds = bounds.copy()
    bound = bounds[dim]
    bounds[dim] = template_bound
    bounds.insert(dim, bound // template_bound if bound else None)
    return bounds


def schedule_memref_linalg(
    op: stream.StreamingRegionOp,
) -> tuple[tuple[AffineMap, ...], tuple[int, ...]]:
    if not op.accelerator or op.accelerator.data not in (
        "snax_alu",
        "snax_gemm",
        "snax_gemmx",
    ):
        raise NotImplementedError(
            "only snax_alu, snax_gemm and snax_gemmx are supported right now"
        )

    # only handle snax_alu and snax_gemm for now
    template = None
    template_bounds = None
    if op.accelerator.data == "snax_alu":
        template = [AffineMap.from_callable(lambda x, y: (4 * x + y,))] * 3
        template_bounds = (None, 4)
    elif op.accelerator.data == "snax_gemm":
        M, N, K, m, n, k = (AffineDimExpr(i) for i in range(6))
        template = [
            AffineMap(6, 0, (M * 8 + m, K * 8 + k)),
            AffineMap(6, 0, (K * 8 + k, N * 8 + n)),
            AffineMap(6, 0, (M * 8 + m, N * 8 + n)),
        ]
        template_bounds = (None, None, None, 8, 8, 8)
    elif op.accelerator.data == "snax_gemmx":
        # TODO: move templates to accelerator definitions
        if isinstance(op.body.block.first_op.body.block.first_op, QMacOp):
            # gemm
            M, N, K, m, n, k = (AffineDimExpr(i) for i in range(6))
            template = [
                AffineMap(6, 0, (M * 8 + m, K * 8 + k)),
                AffineMap(6, 0, (K * 8 + k, N * 8 + n)),
                AffineMap(6, 0, (M * 8 + m, N * 8 + n)),
            ]
            template_bounds = (None, None, None, 8, 8, 8)
        else:
            # rescale function of gemmx
            M, K, m, k = (AffineDimExpr(i) for i in range(4))
            template = [
                AffineMap(4, 0, (M * 8 + m, K * 8 + k)),
                AffineMap(4, 0, (M * 8 + m, K * 8 + k)),
            ]
            template_bounds = (None, None, 8, 8)

    assert template
    assert template_bounds

    schedule = list(pattern.data for pattern in op.patterns.data)
    schedule_bounds: list[int] = list(op.get_static_pattern_bounds())

    for i in range(template[0].num_dims):
        # i = 0: look at the last dimension
        # i = 1: look at the second to last dimension
        template_dim = template[0].num_dims - i - 1
        schedule_dim = schedule[0].num_dims - i - 1
        match = False

        for it in range(schedule_dim + 1):
            # keep rotating the remaining dimensions until we have a match

            template_check = tuple(disable_dims(map, template_dim) for map in template)
            schedule_check = tuple(disable_dims(map, schedule_dim) for map in schedule)

            if template_check == schedule_check:
                match = True
                break

            # else rotate the for loops
            schedule = tuple(rotate_dims(map, schedule_dim + 1) for map in schedule)
            schedule_bounds = rotate_bounds(schedule_bounds, schedule_dim + 1)

        if not match:
            raise RuntimeError("failed to match template and schedule")

        # now, check bounds and design potential transfomration map
        if not (template_bound := template_bounds[template_dim]):
            # nothing to worry about, continue to next dim
            continue

        schedule_bound = schedule_bounds[schedule_dim]

        if schedule_bound < template_bound:
            # need to apply padding
            raise NotImplementedError("padding not supported")
        elif schedule_bound >= template_bound:
            # need to split up the schedule
            assert schedule_bound % template_bound == 0
            schedule = [
                tile_dim(schedule_map, schedule_dim, template_bound)
                for schedule_map in schedule
            ]
            schedule_bounds = tile_bounds(schedule_bounds, schedule_dim, template_bound)
            pass

    return tuple(schedule), tuple(schedule_bounds)
