from itertools import permutations


def get_all_divisors_with_factors(prime_factors: list) -> dict:
    """
    Generate all possible divisors from a list of prime factors along with
    the prime factors used to create each divisor.
    Returns a dict mapping divisor -> list of prime factors used.
    For example, [2, 2, 3] produces {1: [], 2: [2], 3: [3], 4: [2, 2], 6: [2, 3], 12: [2, 2, 3]}.
    """
    if not prime_factors:
        return {1: []}

    # Start with divisor 1 using no prime factors
    divisors = {1: []}

    for prime in prime_factors:
        new_divisors = {}
        for divisor, factors_used in divisors.items():
            new_divisor = divisor * prime
            new_factors = factors_used + [prime]
            new_divisors[new_divisor] = new_factors
        divisors.update(new_divisors)

    return divisors


def remove_prime_factors(prime_factors: list, factors_to_remove: list) -> list:
    """
    Remove specific prime factors from a list of prime factors.
    For example, [2, 2, 3] with [2, 3] removed gives [2].
    """
    remaining = prime_factors.copy()
    for factor in factors_to_remove:
        if factor in remaining:
            remaining.remove(factor)
    return remaining


def try_all_permutations(cache_depth: dict, matrix_sizes: dict):
    """
    Try all permutations of loop orders in the cricital loop level loops (K, M, N)
    and return the possible tilings for each permutation.
    each returned tiling will have exactly 7 loops. (2 temporal level loops, 3 critical loop level loops, 2 cached level loops)

    cache_depth is a dictionary with keys 'K', 'M', 'N' and values the cache depth for that loop.
    matrix_sizes is a dictionary with keys 'K', 'M', 'N' and a list of all prime factors of the matrix size for that loop.
    """
    optional_tilings = []
    for perm in permutations(["K", "M", "N"]):
        optional_tilings.extend(
            level1_optimal_tiling(
                cache_depth=cache_depth, matrix_sizes=matrix_sizes, loop_order=perm
            )
        )
    return optional_tilings


def level1_optimal_tiling(cache_depth: dict, matrix_sizes: dict, loop_order: tuple):
    """
    Determine the 2 cached level loops sizes.
    This functions tries to find every possible tiling for the cached level loops.
    The 2 cached level loops are of dimension loop_order[1] and loop_order[2].
    The sizes of these loops are determined by trying every possible tiling that fits in the cache of depth cache_depth[loop_order[0]].
    """
    critical_cache_depth = cache_depth[loop_order[0]]

    # Get all possible tile sizes with their prime factors
    divisors_inner = get_all_divisors_with_factors(matrix_sizes[loop_order[1]])
    divisors_outer = get_all_divisors_with_factors(matrix_sizes[loop_order[2]])

    valid_tilings = []
    for tile_inner, factors_inner in divisors_inner.items():
        for tile_outer, factors_outer in divisors_outer.items():
            if tile_inner * tile_outer <= critical_cache_depth:
                rest_of_matrix_sizes = {k: v.copy() for k, v in matrix_sizes.items()}

                # Remove prime factors used by the tiles
                rest_of_matrix_sizes[loop_order[1]] = remove_prime_factors(
                    matrix_sizes[loop_order[1]], factors_inner
                )
                rest_of_matrix_sizes[loop_order[2]] = remove_prime_factors(
                    matrix_sizes[loop_order[2]], factors_outer
                )

                rest_of_cache_depth = {
                    loop_order[0]: cache_depth[loop_order[0]]
                    // (tile_inner * tile_outer),
                    loop_order[1]: cache_depth[loop_order[1]] // tile_outer,
                    loop_order[2]: cache_depth[loop_order[2]] // tile_inner,
                }

                valid_tilings += [
                    [
                        (loop_order[1], tile_inner, False),
                        (loop_order[2], tile_outer, False),
                    ]
                    + part_tile
                    for part_tile in level2_optimal_tiling(
                        cache_depth=rest_of_cache_depth,
                        matrix_sizes=rest_of_matrix_sizes,
                        loop_order=loop_order,
                    )
                ]

    return valid_tilings


def level2_optimal_tiling(cache_depth: dict, matrix_sizes: dict, loop_order: tuple):
    """
    By using the middle critical loop level loop, the lowest critical loop level bound is found.
    This function determines the size of the lowest critical loop level loop (loop_order[0]),
    by trying every possible tiling that fits in the cache of depth cache_depth[loop_order[1]].
    """
    critical_cache_depth = cache_depth[loop_order[1]]

    # Get all possible tile sizes for the critical loop dimension
    divisors_critical = get_all_divisors_with_factors(matrix_sizes[loop_order[0]])

    valid_tilings = []
    for tile_critical, factors_critical in divisors_critical.items():
        if tile_critical <= critical_cache_depth:
            rest_of_matrix_sizes = {k: v.copy() for k, v in matrix_sizes.items()}

            # Remove prime factors used by the tile
            rest_of_matrix_sizes[loop_order[0]] = remove_prime_factors(
                matrix_sizes[loop_order[0]], factors_critical
            )

            rest_of_cache_depth = {
                loop_order[0]: cache_depth[loop_order[0]],
                loop_order[1]: cache_depth[loop_order[1]] // tile_critical,
                loop_order[2]: cache_depth[loop_order[2]] // tile_critical,
            }

            valid_tilings += [
                [(loop_order[0], tile_critical, True)] + part_tile
                for part_tile in level3_optimal_tiling(
                    cache_depth=rest_of_cache_depth,
                    matrix_sizes=rest_of_matrix_sizes,
                    loop_order=loop_order,
                )
            ]

    return valid_tilings


def level3_optimal_tiling(cache_depth: dict, matrix_sizes: dict, loop_order: tuple):
    """
    Determine the middle critical loop level loop size.
    The size of this loop is determined by trying every possible tiling that fits in the cache of depth cache_depth[loop_order[2]].
    """
    critical_cache_depth = cache_depth[loop_order[2]]

    # Get all possible tile sizes for the critical loop dimension
    divisors_middle = get_all_divisors_with_factors(matrix_sizes[loop_order[1]])

    valid_tilings = []
    for tile_middle, factors_middle in divisors_middle.items():
        if tile_middle <= critical_cache_depth:
            rest_of_matrix_sizes = {k: v.copy() for k, v in matrix_sizes.items()}

            # Remove prime factors used by the tile
            rest_of_matrix_sizes[loop_order[1]] = remove_prime_factors(
                matrix_sizes[loop_order[1]], factors_middle
            )

            valid_tilings.append(
                [(loop_order[1], tile_middle, True)]
                + temporal_tiling(rest_of_matrix_sizes, loop_order)
            )
    return valid_tilings


def temporal_tiling(matrix_sizes: dict, loop_order: tuple):
    """
    The temporal level loops sizes are determined by the remaining prime factors after the critical loop level loops have been tiled.
    """
    from functools import reduce
    from operator import mul

    def product(factors):
        return reduce(mul, factors, 1) if factors else 1

    return [
        (loop_order[2], product(matrix_sizes[loop_order[2]]), True),
        (loop_order[1], product(matrix_sizes[loop_order[1]]), False),
        (loop_order[0], product(matrix_sizes[loop_order[0]]), False),
    ]


def cost_of_tiling(tiling: list, request_per_streamer: dict):
    """
    Calculate the cost of a tiling as the total tile size (product of all tile sizes).
    """
    input_cost = 1
    weight_cost = 1
    output_cost = 1

    for loop, tile_size, is_critical in reversed(tiling):
        if loop == "N" and is_critical:
            break
        input_cost *= tile_size

    for loop, tile_size, is_critical in reversed(tiling):
        if loop == "M" and is_critical:
            break
        weight_cost *= tile_size

    for loop, tile_size, is_critical in reversed(tiling):
        if loop == "K" and is_critical:
            break
        output_cost *= tile_size

    return (
        input_cost * request_per_streamer["input"]
        + weight_cost * request_per_streamer["weight"]
        + output_cost * request_per_streamer["output"]
    )


def get_best_tiling(cache_depth: dict, matrix_sizes: dict):
    """
    Get the best tiling by trying all permutations and selecting the one with the smallest total tile size.
    """
    all_tilings = try_all_permutations(cache_depth, matrix_sizes)
    best_tiling = min(
        all_tilings,
        key=lambda tiling: cost_of_tiling(tiling, request_per_streamer={"input": 8, "weight": 8, "output": 32}),
    )
    return best_tiling


print(
    get_best_tiling(
        cache_depth={"M": 50, "K": 50, "N": 50},
        matrix_sizes={"M": [7, 10], "K": [3, 7], "N": [5, 9]},
    )
)
# print([cost_of_tiling(tiling, request_per_streamer={"input": 8, "weight": 8, "output": 32}) for tiling in try_all_permutations(
#     cache_depth={"M": 50, "K": 50, "N": 50},
#     matrix_sizes={"M": [7, 10], "K": [3, 7], "N": [5, 9]},
# )])
