from collections.abc import Callable, Iterator, Sequence
from functools import reduce
from math import ceil
import math
from operator import mul
from itertools import permutations

import numpy as np

from snaxc.accelerators.streamers.streamers import HasFixedCache, Streamer, StreamerType
from snaxc.ir.dart.access_pattern import Schedule, Template, SchedulePattern
from snaxc.ir.dart.affine_transform import AffineTransform
from snaxc.ir.dart.cost_models import (
    CostModel,
    EnergyCostModel,
    HardwareLatencyCostModel,
    LatencyCostModel,
    OperandDescriptor,
    OperandKind,
    energy_cost_of_tiling,
    hardware_latency_cost_of_tiling,
    latency_cost_of_tiling,
)
from snaxc.ir.dart.zigzag_cost_model import zigzag_cost_of_tiling


def get_prime_factors(n: int) -> list[int]:
    factors = []
    d = 2
    temp = n
    while d * d <= temp:
        while temp % d == 0:
            factors.append(d)
            temp //= d
        d += 1
    if temp > 1:
        factors.append(temp)
    return factors


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


def split_tiling_to_original_loops(
    tiling: list[tuple[int, int, bool]],
    original_loops_per_ldim: dict[int, list[tuple[int, int, list[int]]]],
) -> list[tuple[int, int, bool]]:
    """
    Split a tiling expressed in logical dimension indices back to
    original loop indices.

    tiling: list of (l_id, size, is_critical) inner->outer
    original_loops_per_ldim: dict l_id -> [(orig_dim_idx, bound, prime_factors)]

    Returns: list of (orig_dim_idx, size, is_critical) inner->outer
    """
    # Track remaining prime factors per original loop (stateful across tiles)
    remaining_factors: dict[int, list[int]] = {}
    for loops in original_loops_per_ldim.values():
        for orig_idx, _bound, pf in loops:
            remaining_factors[orig_idx] = list(pf)

    result: list[tuple[int, int, bool]] = []

    for l_id, size, is_crit in tiling:
        loops_info = original_loops_per_ldim[l_id]

        # Fast path: only one original loop in this logical dim
        if len(loops_info) == 1:
            orig_idx = loops_info[0][0]
            for p in get_prime_factors(size):
                remaining_factors[orig_idx].remove(p)
            result.append((orig_idx, size, is_crit))
            continue

        tile_factors = get_prime_factors(size)

        # Greedily assign each prime factor to the first original loop
        # that still has it available
        assigned: dict[int, list[int]] = {info[0]: [] for info in loops_info}
        for p in tile_factors:
            for orig_idx, _, _ in loops_info:
                if p in remaining_factors[orig_idx]:
                    remaining_factors[orig_idx].remove(p)
                    assigned[orig_idx].append(p)
                    break

        # Build sub-tiles; order within a split doesn't matter
        sub_tiles: list[tuple[int, int, bool]] = []
        for orig_idx, _, _ in loops_info:
            sub_size = reduce(mul, assigned[orig_idx], 1)
            if sub_size > 1:
                sub_tiles.append((orig_idx, sub_size, False))

        # The outermost (last) sub-tile inherits the critical flag
        if is_crit and sub_tiles:
            last = sub_tiles[-1]
            sub_tiles[-1] = (last[0], last[1], True)

        result.extend(sub_tiles)

    return result


def cost_of_tiling(
    tiling: list[tuple[int, int, bool]],
    request_per_streamer: list[int],
    invariance_map: list[set[int]],
):
    """
    Calculate the cost of a tiling as the total access count (energy proxy).
    Kept for backward compatibility. Delegates to energy_cost_of_tiling.

    tiling: list of (dim_idx, size, is_critical), ordered inner → outer.
    """
    return energy_cost_of_tiling(tiling, request_per_streamer, invariance_map)


def _build_l_id_to_template(
    schedule: Schedule,
    template: Template,
    logical_inv_map: dict[int, tuple[bool, ...]],
) -> dict[int, int]:
    """Map each Logical Dimension ID to the template dim it tiles.

    Each L_ID groups temporal dims with the same invariance signature.
    A template dim has the same invariance pattern (which operands have
    all-zero columns) as the temporal dims that tile it.  Match them.
    """
    temporal_dims_count = schedule.num_dims - template.num_dims
    template_sigs: dict[int, tuple[bool, ...]] = {}
    for t in range(template.num_dims):
        template_sigs[t] = tuple(
            np.all(sp.pattern.A[:, temporal_dims_count + t] == 0)
            for sp in schedule
        )

    l_id_to_template: dict[int, int] = {}
    for l_id, l_sig in logical_inv_map.items():
        for t, t_sig in template_sigs.items():
            if l_sig == t_sig:
                l_id_to_template[l_id] = t
                break
        else:
            raise RuntimeError(
                f"L_ID {l_id} with invariance sig {l_sig} does not match "
                f"any template dim signature"
            )
    return l_id_to_template


def _convert_to_template_dims(
    tiling_split: list[tuple[int, int, bool]],
    original_loops_per_ldim: dict[int, list[tuple[int, int, list[int]]]],
    l_id_to_template: dict[int, int],
    inv_map_l_id: list[set[int]],
    num_operands: int,
) -> tuple[list[tuple[int, int, bool]], list[set[int]]]:
    """Convert a split tiling and invariance map from L_ID / orig-idx
    space to template-dim space.

    Returns (tiling_tdim, inv_map_tdim) where both use template dim
    indices consistently.
    """
    # Build orig_idx → template_dim mapping
    orig_to_template: dict[int, int] = {}
    for l_id, loops in original_loops_per_ldim.items():
        tdim = l_id_to_template[l_id]
        for orig_idx, _bound, _pf in loops:
            orig_to_template[orig_idx] = tdim

    # Convert tiling entries
    tiling_tdim = [
        (orig_to_template[orig], size, crit)
        for orig, size, crit in tiling_split
    ]

    # Convert invariance map: L_ID → template_dim
    inv_map_tdim: list[set[int]] = [set() for _ in range(num_operands)]
    for l_id, tdim in l_id_to_template.items():
        for op_idx in range(num_operands):
            if l_id in inv_map_l_id[op_idx]:
                inv_map_tdim[op_idx].add(tdim)

    return tiling_tdim, inv_map_tdim


def _build_operand_descriptors(
    streamers: Sequence[Streamer],
    invariance_map: list[set[int]],
    element_bytes: Sequence[int],
    bank_bits: int = 64,
) -> list[OperandDescriptor]:
    """
    Build OperandDescriptor instances from streamer metadata and the
    invariance map produced by find_optimal_tiling.
    """
    descs: list[OperandDescriptor] = []
    for op_idx, streamer in enumerate(streamers):
        # Determine operand kind from streamer type
        if streamer.type == StreamerType.Reader:
            kind = OperandKind.READER
        elif streamer.type == StreamerType.Writer:
            kind = OperandKind.READER_WRITER
        else:
            kind = OperandKind.READER_WRITER

        # Number of consecutive banks accessed per burst:
        spatial_banks = reduce(mul, streamer.spatial_dims, 1)

        descs.append(OperandDescriptor(
            kind=kind,
            spatial_banks=spatial_banks,
            element_bytes=element_bytes[op_idx],
            invariant_dims=frozenset(invariance_map[op_idx]),
            fixed_cache_depth=streamer.fixed_cache_depth,
        ))
    return descs


def solve_optimal_tiling(
    start_cache_depths: dict[int, int],
    matrix_sizes: dict[int, list[int]],
    loop_order: tuple[int, ...],
    invariance_map: list[set[int]],
) -> list[tuple[int, int, bool]]:
    """
    Recursive function to find the optimal tiling for a specific loop order (permutation of critical dims).
    Returns list of (dim_idx, tile_size, is_critical) from Inner to Outer.

    loop_order: list of critical dimension indices. [Innermost, ..., Outermost]
    """

    # We need to handle the levels recursively.
    # Level structure based on optimal_tiling.py:
    # 1. Cached Level (Innermost): Uses factors of all dims EXCEPT loop_order[0] (innermost critical). Creates N loops.
    # 2. Critical Loop 0: Tile for loop_order[0].
    # 3. Critical Loop 1: Tile for loop_order[1].
    # ...
    # K. Temporal Level: Uses remaining factors.

    # This seems to imply a specific structure:
    # [Cached Tiles...] + [Crit 0 Tile] + [Crit 1 Tile] ... + [Temporal Tiles...]

    # Let's verify constraints:
    # - Cached Tiles should fit in `cache_depth[loop_order[0]]`.
    # - Crit 0 Tile should fit in `cache_depth[loop_order[1]]`.
    # - ...

    # We can implement this by a recursive function that consumes `loop_order`.

    if not loop_order:
        # No critical loops left (should not happen if we started with some).
        # Just return remaining temporal tiling.
        return generate_temporal_tiling(matrix_sizes)

    current_crit_dim = loop_order[0]
    # Cache depth is determined by the *next* critical dimension in the sequence?
    # In optimal_tiling.py: check `tile <= cache_depth[loop_order[next]]`.
    # For the cached level (innermost), correct constraint is determined by `loop_order[0]`.

    # Let's perform the "Cached Level" search first.
    # We want to find tiles for all dims D != current_crit_dim such that their product <= cache_constraints[current_crit_dim].
    # This matches `level1_optimal_tiling`.

    limit = start_cache_depths.get(current_crit_dim, 50)  # Default cache

    # Find all combination of factors from *other* dimensions that fit in limit.
    # We can treat all other dimensions as a single pool of prime factors for this step?
    # No, we must respect dimension boundaries for the result list.

    # Simplified approach for factors: Flatten all *other* dimensions' factors?
    # But we need to yield (dim, size).
    # Since "order does not matter" in cached level, we can just find a valid set of factors.
    # But we need to iterate ALL valid sets to find optimal.

    # To keep it traceable, let's just implement the specific levels as recursive calls.

    return search_cached_level(
        start_cache_depths, matrix_sizes, loop_order, invariance_map
    )


def search_cached_level(
    cache_depths, matrix_sizes, loop_order, invariance_map
) -> list[tuple[int, int, bool]]:
    critical_dim = loop_order[0]
    limit = cache_depths.get(critical_dim, 50)

    # We can tile any dimension except critical_dim.
    other_dims = [d for d in matrix_sizes.keys() if d != critical_dim]

    # Get divisors for all other dims
    # To avoid combinatorial explosion if many dims, we assume 2-3 dims usually.
    # If generic, we can perform a DFS.

    # Result accumulator
    all_tilings = []
    best_tiling = None
    min_cost = float("inf")
    
    # Heuristic: request_per_streamer constants
    # We need to pass this or define it. using constants for now as per prompt.
    request_per_streamer = [8] * len(invariance_map) # Placeholder
    if len(request_per_streamer) >= 3:
         # Assume order C(0), A(1), B(2)? No, invariance map tells us.
         # Just use 8 for all inputs/weights and 32 for output?
         # User said "input": 8, "weight": 8, "output": 32.
         # We need to know which operand is output. Usually last one.
         request_per_streamer[-1] = 32

    # Generator for cached level tiles
    for cached_tiling, used_factors in generate_valid_multidim_factors(
        matrix_sizes, other_dims, limit
    ):
        # Prepare state for next level
        next_matrix_sizes = {
            k: (
                remove_prime_factors(v, used_factors[k])
                if k in used_factors
                else v.copy()
            )
            for k, v in matrix_sizes.items()
        }
        
        # Calculate reduced cache for critical dim:
        # cache[crit] is consumed by the cached_tiling size?
        # In optimal_tiling.py: rest_of_cache_depth[crit] = cache[crit] // (product of tiles)
        tile_size_product = reduce(mul, (x[1] for x in cached_tiling), 1)
        
        next_cache_depths = cache_depths.copy()
        
        # update cache depths for all dimensions that are not invariant to the current loop
        for d_t, s_t, _ in cached_tiling:
            for d in next_cache_depths:
                if d != d_t:
                    next_cache_depths[d] //= s_t

        # Now step into Critical Levels
        suffix = search_critical_levels(
            next_cache_depths, next_matrix_sizes, loop_order, 0
        )
        
        full_tiling = cached_tiling + suffix
        
        if not full_tiling: continue # Should not happen
        all_tilings.append(full_tiling)

        cost = cost_of_tiling(full_tiling, request_per_streamer, invariance_map)
        
        if cost < min_cost:
            min_cost = cost
            best_tiling = full_tiling

    return best_tiling if best_tiling is not None else [], all_tilings

def generate_valid_multidim_factors(matrix_sizes, dims, limit):
    """
    Yields (tiling_list, used_factors_map).
    tiling_list: list of (dim, size, False)
    used_factors_map: dict[dim, list[factors]]
    """
    if not dims:
        yield [], {}
        return

    first_dim = dims[0]
    rest_dims = dims[1:]
    
    divisors = get_all_divisors_with_factors(matrix_sizes[first_dim])
    
    for size, factors in divisors.items():
        if size > limit:
            continue
            
        new_limit = limit // size
        
        for rest_tiling, rest_factors in generate_valid_multidim_factors(matrix_sizes, rest_dims, new_limit):
            current_tiling = []
            if size > 1:
                current_tiling.append((first_dim, size, False))
            current_tiling.extend(rest_tiling)
            
            current_factors = {first_dim: factors}
            current_factors.update(rest_factors)
            
            yield current_tiling, current_factors

def search_critical_levels(cache_depths, matrix_sizes, loop_order, current_idx):
    if current_idx >= len(loop_order):
        return generate_temporal_tiling(matrix_sizes)
    
    dim = loop_order[current_idx]
    
    # Dimension size for this critical level is constrained by the NEXT critical dim's cache
    # If this is the last critical dim (k), what constrains it?
    # Logic in optimal_tiling.py: level3 (last crit) is constrained by cache[loop_order[2]] (itself? No, previous loops)
    # The hierarchy in optimal_tiling.py is:
    # L1 (Cached) constrained by L0 (Crit 1).
    # L2 (Crit 1) constrained by L1 (Crit 2).
    # L3 (Crit 2) constrained by L2 (Crit 3).
    # ...
    # So dim `loop_order[i]` size is constrained by `cache[loop_order[i+1]]`.
    
    next_idx = current_idx + 1
    if next_idx < len(loop_order):
        constraint_dim = loop_order[next_idx]
        limit = cache_depths.get(constraint_dim, float("inf"))
        is_last_tile = False
    else:
        # Outermost critical loop. Not constrained.
        limit = float("inf")
        is_last_tile = True
    
    divisors = get_all_divisors_with_factors(matrix_sizes[dim])
    
    best_tiling = []
    min_cost = float("inf") # Actually greedy per level or global? 
    # Global search via recursion implies we return ALL valid or finding best.
    # search_cached_level (root) does the finding best.
    # Here we should technically propagate up multiple options?
    # But optimal_tiling.py returns `valid_tilings` list and `get_best_tiling` picks min.
    # To avoid huge object passing, let's just return List[List] and let caller pick?
    # Or optimize recursively.
    
    valid_results = []

    if not is_last_tile:
        all_sizes = divisors.items()
    else:
        all_sizes = [(math.prod(matrix_sizes[dim]), matrix_sizes[dim])]  # For last tile, we can choose not to tile (size=1) if it fits constraints.

    for size, factors in all_sizes:
        if size <= limit:
            # We enforce "No dimension of same type in lower loops as innermost loop of critical loops"
            # This check is complex here inside recursion.
            # But "size > 1" tile implies we use it.
            
            # Recurse
            next_matrix_sizes = matrix_sizes.copy()
            next_matrix_sizes[dim] = remove_prime_factors(matrix_sizes[dim], factors)
            
            next_cache_depths = cache_depths.copy()
            
            # Correct Logic:
            for d in next_cache_depths:
                if d != dim and next_cache_depths[d] != float("inf"):
                    next_cache_depths[d] //= size

            result_lists = search_critical_levels(next_cache_depths, next_matrix_sizes, loop_order, next_idx)
            
            # Result is list of tilings (lists).
            for sub_tiling in result_lists:
                 # Current tile
                 current_tile = [(dim, size, True)]
                 valid_results.append(current_tile + sub_tiling)
                 
    return valid_results

def generate_temporal_tiling(matrix_sizes):
    # Returns a list containing one tiling: the rest of factors for all dims
    tiling = []
    for dim, factors in matrix_sizes.items():
        size = reduce(mul, factors, 1)
        if size > 1:
            tiling.append((dim, size, False))
    return [tiling]


def find_optimal_tiling(
    template: Template,
    schedule: Schedule,
    streamers: Sequence[Streamer],
    element_bytes: Sequence[int],
    cost_model_name: str = "latency",
    schedule_idx: int | None = None,
    *,
    num_banks: int = 32,
) -> Schedule:
    """
    Find the optimal tiling for *schedule* on *template* by exploring
    all valid cache-constrained tilings and ranking them with the
    requested cost model.

    Parameters
    ----------
    template : the accelerator template.
    schedule : the backtrack-produced schedule.
    streamers : per-operand streamer descriptors.
    cost_model_name : "latency" (default, minimises TCDM banking conflicts)
        or "energy" (minimises total accesses).
    """
    # 1. Identify Temporal Dims and their Prime Factors
    temporal_dims_count = schedule.num_dims - template.num_dims

    def get_inv_sig(col_idx):
        return tuple(np.all(sp.pattern.A[:, col_idx] == 0) for sp in schedule)

    bounds = schedule[0].bounds

    # Group temporal dimensions by invariance signature
    dim_groups: dict[tuple[bool, ...], list[tuple[int, int]]] = {}
    for i in range(temporal_dims_count):
        sig = get_inv_sig(i)
        if sig not in dim_groups:
            dim_groups[sig] = []
        dim_groups[sig].append((i, bounds[i]))

    # Build Matrix Sizes and Invariance Map
    matrix_sizes: dict[int, list[int]] = {}
    logical_inv_map: dict[int, tuple[bool, ...]] = {}
    original_loops_per_ldim: dict[int, list[tuple[int, int, list[int]]]] = {}

    idx_counter = 0
    for sig, loop_list in dim_groups.items():
        total_size = reduce(mul, (x[1] for x in loop_list), 1)
        matrix_sizes[idx_counter] = get_prime_factors(total_size)
        logical_inv_map[idx_counter] = sig
        original_loops_per_ldim[idx_counter] = [
            (orig_idx, bound, get_prime_factors(bound))
            for orig_idx, bound in loop_list
        ]
        idx_counter += 1

    num_logical = idx_counter

    # Build Invariance Map for cost function: Operand -> Set of Logical Dims
    num_operands = len(schedule)
    inv_map_for_cost: list[set[int]] = [set() for _ in range(num_operands)]
    for l_id in range(num_logical):
        sig = logical_inv_map[l_id]
        for op_idx, is_inv in enumerate(sig):
            if is_inv:
                inv_map_for_cost[op_idx].add(l_id)

    # Build cache depth constraints from streamer fixed caches
    cache_depths: dict[int, int | float] = {}
    critical_dims_pool: set[int] = set()

    for l_id in range(num_logical):
        sig = logical_inv_map[l_id]
        invariant_operands_indices = [i for i, is_inv in enumerate(sig) if is_inv]

        if invariant_operands_indices:
            critical_dims_pool.add(l_id)

            depths = []
            for op_idx in invariant_operands_indices:
                streamer = streamers[op_idx]
                if any(isinstance(opt, HasFixedCache) for opt in streamer.opts):
                    if streamer.fixed_cache_depth > 0:
                        depths.append(streamer.fixed_cache_depth)

            cache_depths[l_id] = min(depths) if depths else float("inf")

    # Build operand descriptors for the latency cost model
    # Energy model uses L_ID-based invariance (tiling is unsplit).
    # Latency/hardware_latency models need template-dim-based invariance
    # (tiling is split and converted to template dims).
    l_id_to_template = _build_l_id_to_template(
        schedule, template, logical_inv_map
    )
    _inv_map_tdim: list[set[int]] = [set() for _ in range(num_operands)]
    for l_id, tdim in l_id_to_template.items():
        for op_idx in range(num_operands):
            if l_id in inv_map_for_cost[op_idx]:
                _inv_map_tdim[op_idx].add(tdim)

    operand_descs_energy = _build_operand_descriptors(
        streamers, inv_map_for_cost, element_bytes
    )
    operand_descs_latency = _build_operand_descriptors(
        streamers, _inv_map_tdim, element_bytes
    )

    # Request-per-streamer heuristic for energy model
    request_per_streamer = [d.spatial_banks for d in operand_descs_energy]

    all_tilings: list[list[tuple[int, int, bool]]] = []

    crit_list = list(critical_dims_pool)

    for perm in permutations(crit_list):
        partial_tilings = search_cached_level_fixed(
            cache_depths, matrix_sizes, perm
        )
        all_tilings.extend(partial_tilings)

    # Reevaluate critical flags for all tilings first
    all_tilings = [
        reevaluate_critical_flags(t, critical_dims_pool, cache_depths)
        for t in all_tilings
    ]

    _template_bounds = tuple(template[0].bounds)

    if schedule_idx is not None:
        if schedule_idx >= len(all_tilings):
            raise ValueError(f"No schedule found at index {schedule_idx}")
        best_tiling = all_tilings[schedule_idx]
        if cost_model_name == "latency":
            # Split logical-dim tiles back to per-original-loop tiles,
            # then convert to template-dim space for the cost model.
            best_tiling_split = split_tiling_to_original_loops(
                best_tiling, original_loops_per_ldim
            )
            tiling_tdim, inv_tdim = _convert_to_template_dims(
                best_tiling_split, original_loops_per_ldim,
                l_id_to_template, inv_map_for_cost, num_operands,
            )
            cost = latency_cost_of_tiling(tiling_tdim, operand_descs_latency, inv_tdim,
                                          template_bounds=_template_bounds,
                                          num_banks=num_banks)
        elif cost_model_name == "hardware_latency":
            best_tiling_split = split_tiling_to_original_loops(
                best_tiling, original_loops_per_ldim
            )
            tiling_tdim, inv_tdim = _convert_to_template_dims(
                best_tiling_split, original_loops_per_ldim,
                l_id_to_template, inv_map_for_cost, num_operands,
            )
            cost = hardware_latency_cost_of_tiling(tiling_tdim, operand_descs_latency, inv_tdim,
                                                   template_bounds=_template_bounds,
                                                   num_banks=num_banks)
        elif cost_model_name == "zigzag":
            best_tiling_split = split_tiling_to_original_loops(
                best_tiling, original_loops_per_ldim
            )
            tiling_tdim, inv_tdim = _convert_to_template_dims(
                best_tiling_split, original_loops_per_ldim,
                l_id_to_template, inv_map_for_cost, num_operands,
            )
            cost = zigzag_cost_of_tiling(tiling_tdim, operand_descs_latency, inv_tdim,
                                         template_bounds=_template_bounds,
                                         num_banks=num_banks)
        else:
            cost = energy_cost_of_tiling(best_tiling, request_per_streamer, inv_map_for_cost)
        print("Predicted Cost for schedule index", schedule_idx, ":", cost)
    else:
        best_tiling = []
        winning_index = -1
        min_cost = float("inf")
        for index, tiling in enumerate(all_tilings):
            if cost_model_name == "latency":
                tiling_split = split_tiling_to_original_loops(tiling, original_loops_per_ldim)
                tiling_tdim, inv_tdim = _convert_to_template_dims(
                    tiling_split, original_loops_per_ldim,
                    l_id_to_template, inv_map_for_cost, num_operands,
                )
                c = latency_cost_of_tiling(tiling_tdim, operand_descs_latency, inv_tdim,
                                           template_bounds=_template_bounds,
                                           num_banks=num_banks)
            elif cost_model_name == "hardware_latency":
                tiling_split = split_tiling_to_original_loops(tiling, original_loops_per_ldim)
                tiling_tdim, inv_tdim = _convert_to_template_dims(
                    tiling_split, original_loops_per_ldim,
                    l_id_to_template, inv_map_for_cost, num_operands,
                )
                c = hardware_latency_cost_of_tiling(tiling_tdim, operand_descs_latency, inv_tdim,
                                                   template_bounds=_template_bounds,
                                                   num_banks=num_banks)
            elif cost_model_name == "zigzag":
                tiling_split = split_tiling_to_original_loops(tiling, original_loops_per_ldim)
                tiling_tdim, inv_tdim = _convert_to_template_dims(
                    tiling_split, original_loops_per_ldim,
                    l_id_to_template, inv_map_for_cost, num_operands,
                )
                c = zigzag_cost_of_tiling(tiling_tdim, operand_descs_latency, inv_tdim,
                                         template_bounds=_template_bounds,
                                         num_banks=num_banks)
            else:
                c = energy_cost_of_tiling(tiling, request_per_streamer, inv_map_for_cost)
            if c < min_cost and c != -1:
                min_cost = c
                best_tiling = tiling
                winning_index = index
        print("Best schedule index:", winning_index)
    best_tiling = [tile for tile in best_tiling if tile[1] > 1]  # Filter out trivial tiles

    # Split logical-dim tiles back to per-original-loop tiles
    best_tiling_split = split_tiling_to_original_loops(
        best_tiling, original_loops_per_ldim
    )
    best_tiling_split = [tile for tile in best_tiling_split if tile[1] > 1]

    return rebuild_schedule(template, schedule, best_tiling_split, dim_groups)

def reevaluate_critical_flags(tiling, critical_dims_pool, cache_depths):
    # Reevaluate which loop is actually the critical loop based on the final tiling, not just the original invariance signatures.
    new_tiling = []
    already_has_critical_loop = {i: False for i in critical_dims_pool}
    for index, (dim_idx, size, _) in reversed(list(enumerate(tiling))):
        is_crit = (fits_in_cache(dim_idx, tiling[:index], cache_depths[dim_idx]) if cache_depths else False) and not already_has_critical_loop[dim_idx]
        if is_crit:
            already_has_critical_loop[dim_idx] = True
        new_tiling.append((dim_idx, size, is_crit))

    return new_tiling[::-1]

def fits_in_cache(dim_idx, tiling, cache_limit):
    # Calculate the product of tile sizes for the given dimension in the tiling
    product = 1
    for d_idx, size, _ in tiling:
        if d_idx != dim_idx:
            product *= size
    return product <= cache_limit


def search_cached_level_fixed(cache_depths, matrix_sizes, loop_order):
    critical_dim = loop_order[0]
    limit = cache_depths.get(critical_dim, 50)
    other_dims = [d for d in matrix_sizes.keys() if d != critical_dim]

    all_tilings = []

    for cached_tiling, used_factors in generate_valid_multidim_factors(matrix_sizes, other_dims, limit):
        next_matrix_sizes = {
            k: (remove_prime_factors(matrix_sizes[k], used_factors[k]) if k in used_factors else matrix_sizes[k].copy())
            for k in matrix_sizes
        }

        next_cache = cache_depths.copy()

        for d_t, s_t, _ in cached_tiling:
            for d in next_cache:
                if d != d_t:
                    next_cache[d] //= s_t

        suffix_options = search_critical_levels(next_cache, next_matrix_sizes, loop_order, 0)

        for suffix in suffix_options:
            full = cached_tiling + suffix
            all_tilings.append(full)

    return all_tilings


def rebuild_schedule(template, old_schedule, tiling, dim_groups):
    """
    Reconstruct a Schedule from a tiling result.

    tiling: [(orig_dim_idx, size, is_crit), ...] ordered inner → outer.
    """
    # Build base vector per original temporal dim
    orig_dim_to_base_vec: dict[int, np.ndarray] = {}
    for _sig, loop_data in dim_groups.items():
        for orig_idx, _bound in loop_data:
            orig_dim_to_base_vec[orig_idx] = np.array(
                [sp.pattern.A[:, orig_idx] for sp in old_schedule]
            )

    spatial_cols = [
        np.array([sp.pattern.A[:, i + (old_schedule.num_dims - template.num_dims)] for sp in old_schedule])
        for i in range(template.num_dims)
    ]
    spatial_bounds = old_schedule[0].bounds[-(template.num_dims):]

    stride_tracker: dict[int, int] = {idx: 1 for idx in orig_dim_to_base_vec}

    schedule_components: list[tuple[int, list[np.ndarray]]] = []

    for orig_idx, size, _is_crit in tiling:
        vecs = []
        base = orig_dim_to_base_vec[orig_idx]
        for op_idx in range(len(old_schedule)):
            vecs.append(base[op_idx] * stride_tracker[orig_idx])
        schedule_components.append((size, vecs))
        stride_tracker[orig_idx] *= size

    # Reverse to get outer → inner
    schedule_components.reverse()

    new_bounds = [x[0] for x in schedule_components] + list(spatial_bounds)

    new_patterns = []
    for op_idx in range(len(old_schedule)):
        temp_cols = [x[1][op_idx] for x in schedule_components]
        spat_cols = [spatial_cols[i][op_idx] for i in range(template.num_dims)]
        all_cols = temp_cols + spat_cols
        if all_cols:
            new_A = np.column_stack(all_cols)
        else:
            new_A = np.zeros((old_schedule[op_idx].pattern.A.shape[0], 0), dtype=int)
        new_patterns.append(SchedulePattern(
            tuple(new_bounds),
            AffineTransform(new_A, old_schedule[op_idx].pattern.b)
        ))

    return Schedule(new_patterns)


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
        template_bound = template[0].bounds[-inner_dims] if inner_dims <= template.num_dims else None
        schedule_bound = candidate_schedule[0].bounds[-inner_dims]

        if template_bound:
            if schedule_bound <= template_bound:
                pass
            elif schedule_bound % template_bound != 0:
                # TODO: imperfect factorization
                continue
            else:
                # tile schedule
                candidate_schedule = candidate_schedule.tile_dim(schedule.num_dims - inner_dims, template_bound)

        # continue with candidate schedule, with an extra inner dim:
        yield from scheduler_backtrack(template, candidate_schedule, inner_dims + 1, extra_checks)


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
    iteration_types: list[bool] = list(map(lambda x: bool(x), np.any(output_schedule != 0, axis=0).tolist()))
    # the first zero should come after the last 1 for output stationary

    # if only reduction, or only parallel, pure otuput stationary is guaranteed
    if not (True in iteration_types and False in iteration_types):
        return True

    first_reduction_idx = iteration_types.index(False)
    last_parallel_idx = len(iteration_types) - 1 - iteration_types[::-1].index(True)

    # last parallel index should come before first reduction idx for pure output stationarity
    return first_reduction_idx > last_parallel_idx


def is_pure_weight_stationary(template: Template, schedule: Schedule):
    """
    Checks whether a schedule, outside of the template, is fully weight
    stationary. This is determined by making sure all parallel dimensions
    precede the reduction dimensions in the weight operand (second operand).
    """
    # Check for zero bounds to avoid accepting invalid schedules
    if any(b == 0 for b in schedule[0].bounds):
        return False

    # fetch the pattern of the weight operand (assumed to be the second operand)
    if len(schedule) < 2:
        return True
    
    weight_schedule = schedule[1].pattern.A
    # do not consider template dims
    weight_schedule = weight_schedule[:, : -template.num_dims]

    # check whether there are any non-zero elements in every column
    # create iteration_types list with False for reduction, True for parallel
    iteration_types: list[bool] = list(map(lambda x: bool(x), np.any(weight_schedule != 0, axis=0).tolist()))
    # the first zero should come after the last 1 for weight stationary

    # if only reduction, or only parallel, pure weight stationary is guaranteed
    if not (True in iteration_types and False in iteration_types):
        return True

    first_reduction_idx = iteration_types.index(False)
    last_parallel_idx = len(iteration_types) - 1 - iteration_types[::-1].index(True)

    # last parallel index should come before first reduction idx for pure weight stationarity
    return first_reduction_idx > last_parallel_idx


def is_output_channel_stationary(template: Template, schedule: Schedule, channel_dim: int) -> bool:
    """
    Checks whether a schedule is output-channel stationary.
    For this, all outputs of a single output channel must be computed
    before moving to the next output channel.
    A 'channel' in this context is defined as the dimension that is
    relevant only to the second dimension of the output operand.
    """
    # fetch the pattern of the 2nd and last operand
    output_schedule = schedule[-1].pattern.A
    # do not consider template dims
    output_schedule = output_schedule[:, : -template.num_dims]

    assert len(output_schedule.shape) > channel_dim, "Output schedule does not have enough dimensions for this check"

    arr = output_schedule[channel_dim, :]

    # There mustn't be a zero before the first non-zero element in the output channel dimension.
    nonzero_indices = np.nonzero(arr)[0]

    if nonzero_indices.size == 0:
        return True  # all elements are zero
    else:
        first_nonzero_idx = nonzero_indices[0]
        result = np.all(arr[:first_nonzero_idx] != 0)
        return bool(result)


def is_memory_flexible_enough(template: Template, schedule: Schedule, element_sizes: Sequence[int]):
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
        temporal = (s.pattern.A[:, 0 : -template.num_dims] % ceil(TCDM_BANK_WIDTH / size)).any(axis=1)
        # is the dimension spatially unrolled?
        spatial = (s.pattern.A[:, -template.num_dims :] == 1).any(axis=1)
        if (False, True) not in zip(temporal, spatial):
            return False
    return True



def scheduler(
    template: Template,
    schedule: Schedule,
    streamers: Sequence[Streamer],
    extra_checks: Sequence[Callable[[Template, Schedule], bool]] = [
        # defaulting to pure output stationary schedules for now
        is_pure_output_stationary,
    ],
    element_bytes: Sequence[int] = (),
    optimal_tiling: bool = False,
    cost_model_name: str = "latency",
    schedule_idx: int | None = None,
    num_banks: int = 32,
) -> Schedule:
    """
    Main scheduling entry point.

    Parameters
    ----------
    cost_model_name : "latency" to minimise TCDM bank conflicts,
        "energy" to minimise total accesses.
    """
    # Default element_bytes to 1 per operand when not provided
    if not element_bytes:
        element_bytes = [1] * len(streamers)

    if schedule_idx is not None:
        iterator = scheduler_backtrack(template, schedule, extra_checks=extra_checks)
        candidate_schedule = next(iterator)

        if optimal_tiling and any(any(isinstance(opt, HasFixedCache) for opt in streamer.opts) for streamer in streamers):
            return find_optimal_tiling(template, candidate_schedule, streamers, element_bytes, cost_model_name, schedule_idx, num_banks=num_banks)
        return candidate_schedule

    result = next(scheduler_backtrack(template, schedule, extra_checks=extra_checks))
    if optimal_tiling and any(any(isinstance(opt, HasFixedCache) for opt in streamer.opts) for streamer in streamers):
        return find_optimal_tiling(template, result, streamers, element_bytes, cost_model_name, num_banks=num_banks)
    return result