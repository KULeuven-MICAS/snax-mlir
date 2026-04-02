#!/usr/bin/env python3
"""
Enumerate all valid schedule indices for a given MLIR input and accelerator
configuration, then compute the latency cost model prediction for each.

This script runs the snaxc compiler pipeline up to (but not including) the
DartSchedulerPass, then manually enumerates all valid schedule indices and
computes their latency costs using the same internal functions that the
scheduler uses.

Usage:
    python enumerate_schedules.py input.mlir -c config.yaml -o schedule_list.json
"""

import json
import sys
import argparse

import yaml
import numpy as np
from typing import cast
from functools import reduce
from operator import mul
from itertools import permutations

from xdsl.dialects import get_all_dialects
from xdsl.dialects import builtin
from xdsl.dialects.builtin import FixedBitwidthType, MemRefType
from xdsl.parser import Parser
from xdsl.passes import PassPipeline

from snaxc.accelerators.acc_context import AccContext
from snaxc.accelerators.snax import SNAXStreamer
from snaxc.accelerators.streamers.streamers import HasFixedCache
from snaxc.dialects import get_all_snax_dialects, dart
from snaxc.ir.dart.access_pattern import Schedule, SchedulePattern
from snaxc.ir.dart.scheduler import (
    scheduler_backtrack,
    is_pure_output_stationary,
    is_memory_flexible_enough,
    _build_operand_descriptors,
    _build_l_id_to_template,
    get_prime_factors,
    search_cached_level_fixed,
    reevaluate_critical_flags,
)
from snaxc.ir.dart.cost_models import hardware_latency_cost_of_tiling, latency_cost_of_tiling
from snaxc.tools.config_parser import parse_config

# Passes to run before the scheduler (same order as snaxc_main.py)
from snaxc.transforms.alloc_to_global import AllocToGlobalPass
from snaxc.transforms.convert_linalg_to_kernel import ConvertLinalgToKernel
from snaxc.transforms.dart.convert_linalg_to_dart import ConvertLinalgToDart
from snaxc.transforms.dart.dart_fuse_operations import DartFuseOperationsPass
from snaxc.transforms.dispatch_kernels import DispatchKernels
from snaxc.transforms.frontend.frontend_transform import FrontendTransformPass
from snaxc.transforms.frontend.preprocess_mlir import PreprocessPass
from snaxc.transforms.fuse_accumulation_memrefs import FuseAccumulationMemrefsPass
from snaxc.transforms.insert_accfg_op import InsertAccOp
from snaxc.transforms.set_memory_space import SetMemorySpace
from snaxc.transforms.snax_bufferize import SnaxBufferize


def load_and_preprocess(mlir_path: str, config_path: str):
    """Load MLIR and run the compiler pipeline up to (but not including)
    DartSchedulerPass."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ctx = parse_config(config)
    ctx.allow_unregistered = True

    # Register all dialects
    all_dialects = get_all_dialects()
    all_dialects.pop("accfg", None)
    all_dialects.pop("stream", None)
    all_dialects.update(get_all_snax_dialects())
    for name, factory in all_dialects.items():
        ctx.register_dialect(name, factory)

    # Parse
    with open(mlir_path) as f:
        module = Parser(ctx, f.read(), mlir_path).parse_module()

    # Build pre-scheduler pipeline (same order as snaxc_main.py)
    pass_list = []
    pass_list.append(FrontendTransformPass())
    pass_list.append(PreprocessPass())
    for acc_name in ctx.registered_accelerator_names:
        pass_list.append(InsertAccOp(acc_name))
    pass_list.append(ConvertLinalgToKernel())
    pass_list.append(DispatchKernels())
    pass_list.append(ConvertLinalgToDart())
    pass_list.append(DartFuseOperationsPass())
    pass_list.append(SnaxBufferize())
    pass_list.append(FuseAccumulationMemrefsPass())
    pass_list.append(AllocToGlobalPass())
    pass_list.append(SetMemorySpace())

    pipeline = PassPipeline(tuple(pass_list))
    module.verify()
    pipeline.apply(ctx, module)
    module.verify()

    return module, ctx


def get_all_tilings(template, schedule, streamers, element_bytes):
    """
    Replicate find_optimal_tiling logic to enumerate ALL valid tilings.

    Returns (all_tilings, operand_descs, inv_map_for_cost) so that costs
    can be computed selectively for specific indices.
    """
    temporal_dims_count = schedule.num_dims - template.num_dims

    def get_inv_sig(col_idx):
        return tuple(np.all(sp.pattern.A[:, col_idx] == 0) for sp in schedule)

    bounds = schedule[0].bounds

    # Group temporal dims by invariance signature
    dim_groups = {}
    for i in range(temporal_dims_count):
        sig = get_inv_sig(i)
        if sig not in dim_groups:
            dim_groups[sig] = []
        dim_groups[sig].append((i, bounds[i]))

    # Build matrix sizes and invariance map
    matrix_sizes = {}
    logical_inv_map = {}
    idx_counter = 0
    for sig, loop_list in dim_groups.items():
        total_size = reduce(mul, (x[1] for x in loop_list), 1)
        matrix_sizes[idx_counter] = get_prime_factors(total_size)
        logical_inv_map[idx_counter] = sig
        idx_counter += 1

    num_logical = idx_counter
    num_operands = len(schedule)

    # Invariance map for cost function
    inv_map_for_cost = [set() for _ in range(num_operands)]
    for l_id in range(num_logical):
        sig = logical_inv_map[l_id]
        for op_idx, is_inv in enumerate(sig):
            if is_inv:
                inv_map_for_cost[op_idx].add(l_id)

    # Cache depth constraints
    cache_depths = {}
    critical_dims_pool = set()
    for l_id in range(num_logical):
        sig = logical_inv_map[l_id]
        inv_ops = [i for i, is_inv in enumerate(sig) if is_inv]
        if inv_ops:
            critical_dims_pool.add(l_id)
            depths = []
            for op_idx in inv_ops:
                streamer = streamers[op_idx]
                if any(isinstance(opt, HasFixedCache) for opt in streamer.opts):
                    if streamer.fixed_cache_depth > 0:
                        depths.append(streamer.fixed_cache_depth)
            cache_depths[l_id] = min(depths) if depths else float("inf")

    # Operand descriptors
    operand_descs = _build_operand_descriptors(streamers, inv_map_for_cost, element_bytes)
    request_per_streamer = [d.spatial_banks for d in operand_descs]

    # Enumerate all tilings across all permutations
    all_tilings = []
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

    # Convert from L_ID space → template-dim space for cost models
    l_id_to_template = _build_l_id_to_template(schedule, template, logical_inv_map)

    # Convert invariance map
    inv_map_tdim: list[set[int]] = [set() for _ in range(num_operands)]
    for l_id, tdim in l_id_to_template.items():
        for op_idx in range(num_operands):
            if l_id in inv_map_for_cost[op_idx]:
                inv_map_tdim[op_idx].add(tdim)

    # Convert all tiling entries: L_ID → template dim
    all_tilings = [
        [(l_id_to_template[l_id], size, crit) for l_id, size, crit in tiling]
        for tiling in all_tilings
    ]

    # Operand descriptors use template-dim based invariance
    operand_descs = _build_operand_descriptors(streamers, inv_map_tdim, element_bytes)

    return all_tilings, operand_descs, inv_map_tdim


def enumerate_schedules(module, ctx):
    """
    Walk the module to find dart.OperationOp, enumerate all valid
    schedule_idx values and their latency costs.

    The scheduler() function, when schedule_idx is set, takes the FIRST
    backtrack result and passes schedule_idx to find_optimal_tiling(),
    which selects all_tilings[schedule_idx]. So schedule_idx indexes into
    the full list of tilings (thousands), not into backtrack results.

    This function mirrors that logic: take the first backtrack result,
    enumerate all tilings, and compute the latency cost for each.
    """
    all_costs = {}

    for op in module.walk():
        if not isinstance(op, dart.OperationOp):
            continue
        if not op.accelerator:
            continue

        accelerator_type = ctx.get_acc(op.accelerator.data)
        if not isinstance(accelerator_type, SNAXStreamer):
            continue

        template = accelerator_type.get_template(op)
        schedule_bounds = tuple(op.get_static_pattern_bounds())
        schedule = Schedule(
            SchedulePattern(schedule_bounds, pattern.data)
            for pattern in op.patterns.data
        )
        schedule = schedule.canonicalize()

        streamers = accelerator_type.get_streamers(op)
        element_sizes = [
            cast(MemRefType[FixedBitwidthType], oper.type).element_type.size
            for oper in op.operands
        ]

        extra_checks = [
            is_pure_output_stationary,
            lambda t, s: is_memory_flexible_enough(t, s, element_sizes),
        ]

        has_fixed_cache = any(
            any(isinstance(opt, HasFixedCache) for opt in s.opts)
            for s in streamers
        )

        if has_fixed_cache:
            # Mirror the scheduler(): take the first backtrack result
            bt_result = next(
                scheduler_backtrack(template, schedule, extra_checks=extra_checks)
            )

            # Enumerate ALL tilings for this backtrack result
            all_tilings, operand_descs, inv_map_for_cost = get_all_tilings(
                template, bt_result, streamers, element_sizes
            )

            print(
                f"  Found {len(all_tilings)} tilings, computing latency costs...",
                file=sys.stderr,
            )

            for idx, tiling in enumerate(all_tilings):
                cost = hardware_latency_cost_of_tiling(
                    tiling, operand_descs, inv_map_for_cost,
                    template_bounds=tuple(template[0].bounds)
                )
                all_costs[idx] = cost
                if cost != -1:
                    print(
                        f"    Cost for tiling {idx}/{len(all_tilings)}: {cost}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"    ERROR IN COST MODEL for tiling {idx}/{len(all_tilings)}",
                        file=sys.stderr,
                    )
                if (idx + 1) % 500 == 0:
                    print(
                        f"    ... computed cost for {idx + 1}/{len(all_tilings)}",
                        file=sys.stderr,
                    )
        else:
            # No optimal tiling => no cost model available;
            # count backtrack results instead
            backtrack_results = list(
                scheduler_backtrack(template, schedule, extra_checks=extra_checks)
            )
            for bt_idx in range(len(backtrack_results)):
                all_costs[bt_idx] = 0.0

        # Only handle the first dart.OperationOp
        break

    return all_costs


def main():
    parser = argparse.ArgumentParser(
        description="Enumerate all valid schedules and compute latency costs"
    )
    parser.add_argument("mlir_file", help="Input MLIR file")
    parser.add_argument(
        "-c", "--config", required=True, help="Accelerator config YAML"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output JSON file"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Include ALL schedules instead of one representative per unique cost",
    )
    args = parser.parse_args()

    module, ctx = load_and_preprocess(args.mlir_file, args.config)
    costs = enumerate_schedules(module, ctx)

    if not args.all:
        # Keep only one representative schedule per unique cost value
        seen_costs: dict[float, int] = {}
        filtered: dict[int, float] = {}
        for idx, cost in sorted(costs.items()):
            if cost not in seen_costs:
                seen_costs[cost] = idx
                filtered[idx] = cost
        n_before = len(costs)
        costs = filtered
        print(
            f"Deduplicated: {n_before} -> {len(costs)} schedules "
            f"({len(costs)} unique cost values)",
            file=sys.stderr,
        )

    output_data = {
        "num_schedules": len(costs),
        "costs": {str(k): v for k, v in sorted(costs.items())},
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Output {len(costs)} schedules to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
