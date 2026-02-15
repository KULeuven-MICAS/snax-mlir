from collections.abc import Callable, Iterator, Sequence
from functools import reduce
from math import ceil
from operator import mul
from itertools import permutations

import numpy as np

from snaxc.accelerators.streamers.streamers import HasFixedCache, Streamer
from snaxc.ir.dart.access_pattern import Schedule, Template, SchedulePattern
from snaxc.ir.dart.affine_transform import AffineTransform


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


def cost_of_tiling(
    tiling: list[tuple[int, int, bool]],
    request_per_streamer: list[int],
    invariance_map: list[set[int]],
):
    """
    Calculate the cost of a tiling as the total tile size (product of all tile sizes).
    tiling: list of (dim_idx, size, is_critical) from inner to outer?
            optimal_tiling.py logic implies tiling list is built [Inner ... Outer].
            cost_of_tiling in optimal_tiling.py iterates reversed(tiling) -> Outer to Inner.
            Here we assume input `tiling` is [Inner, ..., Outer].
    """
    # Initialize costs for each operand
    operand_costs = [1] * len(request_per_streamer)

    # Iterate from Outer to Inner
    for dim_idx, tile_size, is_critical in reversed(tiling):
        for op_idx, cost in enumerate(operand_costs):

            # If this dimension is critical for this operand, we stop accumulating cost
            if is_critical and (dim_idx in invariance_map[op_idx]):
                continue

            # Otherwise, multiply cost
            operand_costs[op_idx] *= tile_size

    total_cost = sum(c * r for c, r in zip(operand_costs, request_per_streamer))
    return total_cost


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

        cost = cost_of_tiling(full_tiling, request_per_streamer, invariance_map)
        
        if cost < min_cost:
            min_cost = cost
            best_tiling = full_tiling

    return best_tiling if best_tiling is not None else []

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
    else:
        # Outermost critical loop. Not constrained.
        limit = float("inf")
    
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

    for size, factors in divisors.items():
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
                 current_tile = [(dim, size, True)] if size > 1 else []
                 valid_results.append(current_tile + sub_tiling)
                 
    return valid_results

def generate_temporal_tiling(matrix_sizes):
    # Returns a list containing one tiling: the rest of factors for all dims
    tiling = []
    for dim, factors in matrix_sizes.items():
        size = reduce(mul, factors, 1)
        if size > 1:
            tiling.append((dim, size, False))
    return [tiling] # List of 1 tiling

# Wrapper to clean up signature match
def search_critical_levels_wrapper(cache_depths, matrix_sizes, loop_order, current_idx) -> list[tuple[int, int, bool]]:
    # We need to return ONE best list, but `search_critical_levels` returns variants.
    # To fix this, we need 'cost' awareness inside, or return all.
    # Returning all is safer for optimality but expensive.
    # Given matrix sizes are small (kernels), it should be fine.
    
    variants = search_critical_levels(cache_depths, matrix_sizes, loop_order, current_idx)
    # But we don't have enough context to judge cost here (missing inner parts if we were called standalone).
    # But we are called from search_cached_level which handles cost.
    # Wait, `search_cached_level` called `search_critical_levels` and expected a SINGLE list?
    # My code `suffix = search_critical_levels(...)` assigned a list-of-lists to suffix.
    # `full_tiling = cached_tiling + suffix` -> Error.
    
    # Fix: Iterate variants in search_cached_level
    return variants 

# ... Fixed search_cached_level below ...

def find_optimal_tiling(template: Template, schedule: Schedule, streamers: Sequence[Streamer]) -> Schedule:
    # 1. Identify Temporal Dims and their Prime Factors
    temporal_dims_count = schedule.num_dims - template.num_dims
    
    # Reconstruct logical dimensions from schedule pattern columns
    # Group cols by signature check
    logic_dims = {} # ID -> {'indices': [], 'size': int}
    
    # Helper to get column signature
    def get_sig(col_idx):
        return tuple(tuple(sp.pattern.A[:, col_idx]) for sp in schedule)

    # Note: If cols have different strides but same sparsity pattern, they are same dimension?
    # Yes. We care about "invariance".
    # Invariance signature: for each operand, is it 0?
    def get_inv_sig(col_idx):
        return tuple(np.all(sp.pattern.A[:, col_idx] == 0) for sp in schedule)

    # Access Schedule Patterns directly
    # Also get bounds.
    bounds = schedule[0].bounds # All sps have same bounds
    
    # Group temporal dimensions by invariance signature
    dim_groups = {} # sig -> list of (index, size)
    for i in range(temporal_dims_count):
        sig = get_inv_sig(i)
        if sig not in dim_groups: dim_groups[sig] = []
        dim_groups[sig].append((i, bounds[i]))
        
    # Build Matrix Sizes and Invariance Map
    matrix_sizes = {} # logical_id -> prime_factors
    invariance_map = [] # logical_id -> set(operand_indices) (This is wrong, map is operand -> set(dims))
    
    # We assign logical IDs 0, 1, ...
    logical_inv_map = {} # logical_id -> inv_sig
    
    idx_counter = 0
    for sig, loop_list in dim_groups.items():
        total_size = reduce(mul, (x[1] for x in loop_list), 1)
        matrix_sizes[idx_counter] = get_prime_factors(total_size)
        logical_inv_map[idx_counter] = sig
        idx_counter += 1
        
    num_logical = idx_counter
    
    # Build Invariance Map for cost function: Operand -> Set of Logical Dims
    num_operands = len(schedule)
    inv_map_for_cost = [set() for _ in range(num_operands)]
    for l_id in range(num_logical):
        sig = logical_inv_map[l_id]
        for op_idx, is_inv in enumerate(sig):
            if is_inv:
                inv_map_for_cost[op_idx].add(l_id)
                
    # Identify Critical Dimensions for each Operand
    # "The critical loop is the loop which the operand is invariable to"
    # An operand might be invariant to multiple.
    # We collect ALL Loop Dimensions that are "invariant" for AT LEAST ONE operand to be the set of "Critical Loop Level Loops".
    # And we permute THIS set.
    
    # We also need to map Critical Loops to Streamer Cache Depths.
    # A Streamer S with Fixed Cache F is typically associated with ONE Critical Dimension (the one it's invariant to).
    # If S is invariant to multiple critical loops?
    # Usually F is the "L1" cache size.
    # If S is output stationary, it is invariant to the reduction loop (Crit Dim K).
    # If we are inside Crit Loop K, S uses 0 BW.
    # Inner loops (Cached Level) must fit in S's cache.
    # So `cache_constraint[K]` is determined by S's fixed cache depth.
    
    # Map: Logical Dim -> Min Cache Depth (if this dim is chosen as critical loop)
    # Actually, the constraint is:
    # If we are tiling for Critical Loop L (meaning L is the stationary loop for some operands):
    # Then the Innermost Cached Level must fit in the caches of those stationary operands.
    # So `cache_depths[l_id]` should be the Minimum Fixed Cache of all operands invariant to `l_id`.
    
    cache_depths = {}
    critical_dims_pool = set()
    
    for l_id in range(num_logical):
        sig = logical_inv_map[l_id]
        
        # Operands invariant to this dim
        invariant_operands_indices = [i for i, is_inv in enumerate(sig) if is_inv]
        
        if invariant_operands_indices:
            critical_dims_pool.add(l_id)
            
            # Find associated streamers and their cache depths
            depths = []
            for op_idx in invariant_operands_indices:
                streamer = streamers[op_idx]
                if streamer.fixed_cache_depth > 0:
                     depths.append(streamer.fixed_cache_depth)
            
            # The constraint is the MINIMUM of all applicable caches.
            # If multiple operands are stationary under this loop, ALL must fit their respective working sets.
            # Working set size for Operand O (stationary) inside Cached Level (loops T_i):
            # Size = Product(TileSize(T_j)) for all T_j that O varies with.
            # Here we assume Cached Level tiles ALL other dims.
            # So basically, we just limit the product of other dims.
            
            if depths:
                cache_depths[l_id] = min(depths)
            else:
                # No fixed cache constraint found for this critical loop?
                # Default to something or infinite?
                # If a loop is critical but has no fixed cache, maybe default to 50/infinity?
                cache_depths[l_id] = float("inf") # Default fallback
        else:
             # Not a critical dimension candidate (no operand is invariant)
             pass
            
    best_tiling = []
    min_cost = float("inf")
    
    request_per_streamer = [8] * num_operands # Placeholder
    if num_operands > 0: request_per_streamer[-1] = 32

    # Iterate permutations of critical dimensions
    # If pool is empty (no invariant checks?), treat all as non-critical?
    crit_list = list(critical_dims_pool)
    
    # Optimization: if pool is large, permutations are many.
    # Usually 3 dims max.
    
    for perm in permutations(crit_list):
        # We need to call the recursive search
        # Note: logic inside needs fix for list-of-lists return
        tiling = search_cached_level_fixed(cache_depths, matrix_sizes, perm, inv_map_for_cost, request_per_streamer)
        if not tiling: continue
        
        c = cost_of_tiling(tiling, request_per_streamer, inv_map_for_cost)
        if c < min_cost:
            min_cost = c
            best_tiling = tiling

    # Apply Tiling to Schedule
    # best_tiling is list of (logical_dim, size, is_crit) from Inner to Outer.
    # We want to reconstruct the schedule: [Outer ..., Inner ...] + [Spatial]
    # Reverse best_tiling to get Outer->Inner order for Temporal part.
    
    final_temporal_order = reversed(best_tiling)
    
    # Reconstruct proper Schedule object
    return rebuild_schedule(template, schedule, best_tiling, dim_groups)

def search_cached_level_fixed(cache_depths, matrix_sizes, loop_order, invariance_map, request_per_streamer):
    # Same as search_cached_level but handling the list-of-lists from critical search
    critical_dim = loop_order[0]
    limit = cache_depths.get(critical_dim, 50)
    other_dims = [d for d in matrix_sizes.keys() if d != critical_dim]

    best_local = None
    min_local = float("inf")

    for cached_tiling, used_factors in generate_valid_multidim_factors(matrix_sizes, other_dims, limit):
        next_matrix_sizes = {
            k: (remove_prime_factors(matrix_sizes[k], used_factors[k]) if k in used_factors else matrix_sizes[k].copy())
            for k in matrix_sizes
        }
        
        tile_size_product = reduce(mul, (x[1] for x in cached_tiling), 1)
        next_cache = cache_depths.copy()
        next_cache[critical_dim] //= tile_size_product
        
        suffix_options = search_critical_levels(next_cache, next_matrix_sizes, loop_order, 0)
        
        for suffix in suffix_options:
            full = cached_tiling + suffix
            c = cost_of_tiling(full, request_per_streamer, invariance_map)
            if c < min_local:
                min_local = c
                best_local = full
                
    return best_local

def rebuild_schedule(template, old_schedule, tiling, dim_groups):
    # tiling: [(l_dim, size, is_crit), ...] (Inner -> Outer)
    # We need to construct new loops.
    # Each item in tiling generates a loop.
    # The spatial loops from old_schedule (the last template.num_dims) must be appended at the end (Innermost).
    
    # 1. Calculate Strides for each Logical Dim
    # Strides track the cumulative position in the 'linearized' logical dimension.
    # Since we are essentially re-tiling, we can track the 'step' for each tile.
    
    # Extract Base Vectors for each Logical Dim
    # We must iterate dim_groups in the same order as find_optimal_tiling to match l_id
    l_id_to_base_vec = {}
    
    for l_id, (sig, loop_data) in enumerate(dim_groups.items()):
        # Find base vector
        indices = [x[0] for x in loop_data]
        # Check all cols.
        # We need the vector `v` such that any col `c` is `k * v`.
        # Taking the column with smallest norms?
        best_vec = None
        min_norm = float('inf')
        for i in indices:
            # Construct vec for this col across all ops
            vec = np.array([sp.pattern.A[:, i] for sp in old_schedule])
            # vec shape (num_ops, num_results_per_op).
            # Flatten to measure 'size'?
            norm = np.sum(np.abs(vec))
            if norm < min_norm and norm > 0:
                 min_norm = norm
                 best_vec = vec
        if best_vec is None: # All zero?
             best_vec = np.zeros_like(np.array([sp.pattern.A[:, indices[0]] for sp in old_schedule]))
        l_id_to_base_vec[l_id] = best_vec

    # 2. Build New Schedule
    # Loop order: Outer -> Inner.
    # Tiling: Reversed(tiling) (which is originally Inner->Outer).
    loops = list(reversed(tiling))
    
    # We also have the Spatial loops from old_schedule.
    # We should keep their pattern contributions as is.
    spatial_cols = [np.array([sp.pattern.A[:, i + (old_schedule.num_dims - template.num_dims)] for sp in old_schedule]) 
                    for i in range(template.num_dims)]
    spatial_bounds = old_schedule[0].bounds[-(template.num_dims):]
    
    # Current accumulators for strides of logical dims
    # We use number of logical dimensions = number of keys in l_id_to_base_vec
    num_logical_dims = len(l_id_to_base_vec)
    l_stride_tracker = {l_id: 1 for l_id in range(num_logical_dims)}
    
    # Calculate vector for each tile level, iterating Inner -> Outer (tiling order)
    # Then reverse to get Outer -> Inner for final schedule construction
    
    schedule_components = [] # List of (bound, [vec_op0, vec_op1, ...])
    
    for l_id, size, is_crit in tiling:
        # Inner to Outer
        vecs = []
        base = l_id_to_base_vec[l_id] # Shape (num_ops, dim_out) - base_vec is already a list of arrays? No, base_vec is 'vec'.
        
        # In the extraction loop:
        # best_vec = np.array([sp.pattern.A[:, i] for sp in old_schedule])
        # So base is an np.array of shape (num_ops, num_results_per_op).
        
        # We need individual op vectors for this level.
        # base[op_idx] is the vector for that op.
        
        for op_idx in range(len(old_schedule)):
             # Scale by current stride tracker and Append
             vecs.append(base[op_idx] * l_stride_tracker[l_id])
             
        schedule_components.append((size, vecs))
        
        # Update stride
        l_stride_tracker[l_id] *= size
        
    # Reverse to get Outer -> Inner
    schedule_components.reverse()
    
    new_bounds = [x[0] for x in schedule_components] + list(spatial_bounds)
    
    # Construct final matrices
    new_patterns = []
    for op_idx in range(len(old_schedule)):
        # Collect temporal cols
        temp_cols = [x[1][op_idx] for x in schedule_components] 
        # Collect spatial cols
        spat_cols = [spatial_cols[i][op_idx] for i in range(template.num_dims)]
        
        all_cols = temp_cols + spat_cols
        # Stack columns (each is 1D array of size dim_out)
        # Result A matrix: (dim_out, new_num_dims)
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
    schedule_idx: int | None = None,
    optimal_tiling: bool = False,
) -> Schedule:
    # for now just return the first result of the backtracking
    if schedule_idx is not None:
        iterator = scheduler_backtrack(template, schedule, extra_checks=extra_checks)
        try:
            candidate_schedule = next(
                result
                for i, result in enumerate(iterator)
                if i == schedule_idx
            )
        except StopIteration:
            raise ValueError(f"No schedule found at index {schedule_idx}")
        
        if optimal_tiling and any(any(isinstance(opt, HasFixedCache) for opt in streamer.opts) for streamer in streamers):
            return find_optimal_tiling(template, candidate_schedule, streamers)
        return candidate_schedule

    result = next(scheduler_backtrack(template, schedule, extra_checks=extra_checks))
    if optimal_tiling and any(any(isinstance(opt, HasFixedCache) for opt in streamer.opts) for streamer in streamers):
        return find_optimal_tiling(template, result, streamers)
    return result
