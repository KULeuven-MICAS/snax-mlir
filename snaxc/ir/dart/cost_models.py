"""
Cost models for the SNAX DART scheduler.

This module provides cost model implementations for evaluating tiling strategies:
- EnergyCostModel: Naive model minimizing total TCDM accesses (proxy for energy).
- LatencyCostModel: Cycle-accurate model minimizing banking conflicts (proxy for latency).
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import reduce
from math import gcd
from operator import mul
from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Streamer / operand metadata needed by the cost models
# ---------------------------------------------------------------------------

class OperandKind(Enum):
    """Whether the operand is read, written, or both."""
    READER = auto()
    WRITER = auto()
    READER_WRITER = auto()


@dataclass(frozen=True)
class OperandDescriptor:
    """
    Lightweight description of a single operand's access behaviour that is
    independent of any particular tiling.

    Attributes:
        kind: READER, WRITER, or READER_WRITER.
        spatial_banks: Number of consecutive TCDM banks accessed per burst.
            Equal to (product of spatial dims)
        invariant_dims: Set of *logical* dimension ids to which the operand
            is invariant (i.e. the stride is 0 for those dims).
    """
    kind: OperandKind
    spatial_banks: int
    invariant_dims: frozenset[int]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CostModel(ABC):
    """Base interface for tiling cost models."""

    @abstractmethod
    def cost(
        self,
        tiling: list[tuple[int, int, bool]],
        operand_descriptors: Sequence[OperandDescriptor],
        request_per_streamer: Sequence[int],
        invariance_map: list[set[int]],
        *,
        num_banks: int = 32,
        bank_bits: int = 64,
    ) -> float:
        """
        Evaluate the cost of *tiling*.

        Parameters
        ----------
        tiling : list of (logical_dim_idx, tile_size, is_critical)
            Ordered **inner → outer**.
        operand_descriptors : per-operand metadata.
        request_per_streamer : per-operand requests weight (used by energy model).
        invariance_map : operand → set of logical dims it is invariant to.
        num_banks : number of TCDM banks.
        bank_bits : bit-width of a single bank word (default 64).
        """
        ...


# ---------------------------------------------------------------------------
# Energy cost model (the *old* naive model preserved as-is)
# ---------------------------------------------------------------------------

class EnergyCostModel(CostModel):
    """
    Naive cost model that estimates energy by summing weighted operand working-set
    sizes.  This is the original ``cost_of_tiling`` logic.
    """

    def cost(
        self,
        tiling: list[tuple[int, int, bool]],
        operand_descriptors: Sequence[OperandDescriptor],
        request_per_streamer: Sequence[int],
        invariance_map: list[set[int]],
        *,
        num_banks: int = 32,
        bank_bits: int = 64,
    ) -> float:
        return energy_cost_of_tiling(tiling, list(request_per_streamer), invariance_map)


def energy_cost_of_tiling(
    tiling: list[tuple[int, int, bool]],
    request_per_streamer: list[int],
    invariance_map: list[set[int]],
) -> float:
    """
    Original cost function: total TCDM accesses weighted by streamer request
    counts.  Kept for energy-consumption estimation.

    *tiling* is ordered **inner → outer**.
    """
    operand_costs = [1] * len(request_per_streamer)
    # iterate outer → inner
    for dim_idx, tile_size, is_critical in reversed(tiling):
        for op_idx, _ in enumerate(operand_costs):
            if is_critical and (dim_idx in invariance_map[op_idx]):
                continue
            operand_costs[op_idx] *= tile_size
    return float(sum(c * r for c, r in zip(operand_costs, request_per_streamer)))


# ---------------------------------------------------------------------------
# Latency cost model – banking-conflict aware
# ---------------------------------------------------------------------------

class LatencyCostModel(CostModel):
    """
    Cycle-accurate(ish) cost model that counts the total number of TCDM cycles
    by simulating banking conflicts.

    Key optimisations
    -----------------
    * Instead of computing absolute addresses we work with **stride residues**
      modulo *num_banks*.  Two bursts from different operands collide iff
      their bank-sets overlap, which can be checked purely from residues and
      burst widths.
    * For the innermost loop we detect *steady-state* conflict patterns and
      multiply by the iteration count rather than iterating element-by-element.
    * Reader/Writer timing offsets (readers fire at counter == 0 for the
      critical + invariant loops; writers fire at counter == bound-1;
      reader-writers are offset by 2 iterations) are faithfully modelled.
    """

    def cost(
        self,
        tiling: list[tuple[int, int, bool]],
        operand_descriptors: Sequence[OperandDescriptor],
        request_per_streamer: Sequence[int],
        invariance_map: list[set[int]],
        *,
        num_banks: int = 32,
        bank_bits: int = 64,
    ) -> float:
        return latency_cost_of_tiling(
            tiling,
            operand_descriptors,
            invariance_map,
            num_banks=num_banks,
            bank_bits=bank_bits,
        )


# ---------------------------------------------------------------------------
# Helper: bank-set overlap check
# ---------------------------------------------------------------------------

def _bank_sets_overlap(
    start_a: int, width_a: int, start_b: int, width_b: int, num_banks: int
) -> int:
    """
    Return the number of overlapping banks between two contiguous burst
    ranges on a *num_banks*-wide modular bank space.

    Each burst occupies banks ``[start % num_banks, start % num_banks + width)``.
    Since widths are typically ≤ num_banks and the space wraps, we check
    modular interval overlap.
    """
    if width_a == 0 or width_b == 0:
        return 0

    a0 = start_a % num_banks
    b0 = start_b % num_banks

    # Expand both ranges into sorted sets of bank indices (mod num_banks).
    # For small widths (≤ 64 typically) this is efficient.
    set_a = set((a0 + i) % num_banks for i in range(width_a))
    set_b = set((b0 + i) % num_banks for i in range(width_b))
    return len(set_a & set_b)


def _count_max_bank_hits(
    bank_starts: list[int],
    burst_widths: list[int],
    num_banks: int,
) -> int:
    """
    Given a list of burst accesses (one per active operand in this cycle),
    each touching ``burst_widths[i]`` consecutive banks starting at
    ``bank_starts[i] % num_banks``, return the maximum number of accesses
    hitting any single bank.
    """
    if not bank_starts:
        return 0

    bank_counts: dict[int, int] = {}
    for start, width in zip(bank_starts, burst_widths):
        base = start % num_banks
        for offset in range(width):
            b = (base + offset) % num_banks
            bank_counts[b] = bank_counts.get(b, 0) + 1
    return max(bank_counts.values()) if bank_counts else 0


# ---------------------------------------------------------------------------
# Stride computation helpers
# ---------------------------------------------------------------------------

def _compute_operand_stride_per_tile(
    tiling: list[tuple[int, int, bool]],
    operand_descriptors: Sequence[OperandDescriptor],
    invariance_map: list[set[int]],
    bank_bits: int,
) -> list[list[int]]:
    """
    For every operand, compute the *byte stride* contributed by each tiling
    level.

    The stride for operand ``op`` at tiling level ``lvl`` =
    ``base_stride[op] * cumulative_multiplier[logical_dim]``
    where ``base_stride`` comes from the spatial-bank count and bank width.

    Since we only need bank indices (address // bank_bytes % num_banks is
    equivalent to (address // bank_bytes) % num_banks), we return strides
    in units of **bank words** (each bank_bits / 8 bytes).

    For an invariant dimension the stride is 0.

    Returns
    -------
    strides : list[list[int]]
        ``strides[op_idx][level_idx]`` – stride in bank-word units for
        that operand at that tiling level.
    """
    bank_bytes = bank_bits // 8
    num_ops = len(operand_descriptors)
    num_levels = len(tiling)

    # For each logical dim, track cumulative factor (inner → outer)
    # tiling is already inner → outer.
    dim_cumulative: dict[int, int] = {}

    strides: list[list[int]] = [[] for _ in range(num_ops)]

    for lvl, (dim_idx, tile_size, _is_crit) in enumerate(tiling):
        cum = dim_cumulative.get(dim_idx, 1)
        for op_idx, desc in enumerate(operand_descriptors):
            if dim_idx in desc.invariant_dims:
                strides[op_idx].append(0)
            else:
                # base element stride = spatial_banks * bank_bytes (bytes per
                # spatial burst). In bank-word units that is just spatial_banks.
                # But the *temporal* stride for this dim is
                # spatial_banks * cum (in bank-word units).
                strides[op_idx].append(desc.spatial_banks * cum)
        dim_cumulative[dim_idx] = cum * tile_size

    return strides


# ---------------------------------------------------------------------------
# Access-activity predicates
# ---------------------------------------------------------------------------

def _reader_active(
    counters: list[int],
    tiling: list[tuple[int, int, bool]],
    invariant_dims: frozenset[int],
) -> bool:
    """
    A reader fires when, for every tiling level at or below the outermost
    critical level whose dimension the operand is invariant to (stride = 0),
    the counter is 0.
    """
    # Find the outermost critical level (regardless of invariance).
    critical_level: int | None = None
    for lvl in range(len(tiling) - 1, -1, -1):  # outer → inner
        _, _, is_crit = tiling[lvl]
        if is_crit:
            critical_level = lvl
            break

    if critical_level is None:
        # No critical loop → always active
        return True

    # All levels from critical_level down to 0 that the operand is invariant
    # to must have counter == 0.
    for lvl in range(critical_level, -1, -1):
        dim_idx = tiling[lvl][0]
        if dim_idx in invariant_dims:
            if counters[lvl] != 0:
                return False
    return True


def _writer_active(
    counters: list[int],
    bounds: list[int],
    tiling: list[tuple[int, int, bool]],
    invariant_dims: frozenset[int],
) -> bool:
    """
    A writer fires when, for every tiling level at or below the outermost
    critical level whose dimension the operand is invariant to, the counter
    is at its last value (bound - 1).
    """
    # Find the outermost critical level (regardless of invariance).
    critical_level: int | None = None
    for lvl in range(len(tiling) - 1, -1, -1):
        _, _, is_crit = tiling[lvl]
        if is_crit:
            critical_level = lvl
            break

    if critical_level is None:
        return True

    for lvl in range(critical_level, -1, -1):
        dim_idx = tiling[lvl][0]
        if dim_idx in invariant_dims:
            if counters[lvl] != bounds[lvl] - 1:
                return False
    return True


# ---------------------------------------------------------------------------
# (The old _analyse_innermost_period was removed – its functionality is now
#  subsumed by the global-step simulation in _simulate_nested.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Differential / modular fast-path for steady-state detection
# ---------------------------------------------------------------------------

def _compute_period_cost_fast(
    strides_mod: list[int],
    burst_widths: list[int],
    inner_stride_mod: list[int],
    inner_bound: int,
    num_banks: int,
) -> int | None:
    """
    Attempt a fast analytical cost computation for the innermost loop.

    If all operands are always active (no gating) we can compute the
    per-iteration conflict pattern which repeats with period
    ``lcm(inner_strides_mod) | num_banks``.

    Returns total cycles or None if the fast path is not applicable.
    """
    # All strides are already mod num_banks
    # The conflict pattern of the innermost loop repeats every
    # P = num_banks / gcd(all non-zero inner strides, num_banks) iterations.

    nonzero_strides = [s for s in inner_stride_mod if s != 0]
    if not nonzero_strides:
        # All strides zero → every iteration accesses same banks
        hits = _count_max_bank_hits(strides_mod, burst_widths, num_banks)
        if hits == 0:
            return 0
        return max(hits, 1) * inner_bound

    g = nonzero_strides[0]
    for s in nonzero_strides[1:]:
        g = gcd(g, s)
    g = gcd(g, num_banks)
    period = num_banks // g

    if period > inner_bound:
        # Period longer than loop → no shortcut, fall back
        return None

    # Simulate one period and multiply
    per_period_cycles = 0
    for ic in range(period):
        starts = [(s + ist * ic) % num_banks for s, ist in zip(strides_mod, inner_stride_mod)]
        hits = _count_max_bank_hits(starts, burst_widths, num_banks)
        per_period_cycles += max(hits, 1)

    full_periods = inner_bound // period
    remainder = inner_bound % period

    total = full_periods * per_period_cycles

    # Handle remainder
    for ic in range(remainder):
        starts = [(s + ist * ic) % num_banks for s, ist in zip(strides_mod, inner_stride_mod)]
        hits = _count_max_bank_hits(starts, burst_widths, num_banks)
        total += max(hits, 1)

    return total


# ---------------------------------------------------------------------------
# Main latency cost function
# ---------------------------------------------------------------------------

def latency_cost_of_tiling(
    tiling: list[tuple[int, int, bool]],
    operand_descriptors: Sequence[OperandDescriptor],
    invariance_map: list[set[int]],
    *,
    num_banks: int = 32,
    bank_bits: int = 64,
) -> float:
    """
    Compute the total number of TCDM cycles for a given tiling by
    simulating banking conflicts.

    Parameters
    ----------
    tiling : list of (dim_idx, tile_size, is_critical)
        Ordered **inner → outer**.
    operand_descriptors : per-operand metadata.
    invariance_map : operand_idx → set of logical dims it is invariant to.
    num_banks : TCDM bank count.
    bank_bits : width of a single TCDM bank word in bits.

    Returns
    -------
    float  – total cycles.
    """
    if not tiling:
        return 0.0

    # Compute per-operand strides in bank-word units
    strides_bank = _compute_operand_stride_per_tile(
        tiling, operand_descriptors, invariance_map, bank_bits
    )

    return float(_simulate_nested(
        tiling, operand_descriptors, invariance_map, strides_bank, num_banks
    ))


def _simulate_nested(
    tiling: list[tuple[int, int, bool]],
    operand_descriptors: Sequence[OperandDescriptor],
    invariance_map: list[set[int]],
    strides_bank: list[list[int]],
    num_banks: int,
) -> int:
    """
    Simulate all iteration steps of the deeply nested for-loop given by
    *tiling*, computing cycle costs including banking conflicts and
    ReaderWriter stalls.

    The iteration steps are enumerated globally (as a mixed-radix counter
    over all tiling levels, innermost changing fastest).  The writer part
    of a ReaderWriter operand is always 2 *global* iteration steps behind
    the reader part.  Every iteration step costs at least 1 cycle (the
    accelerator always executes even when no memory is accessed).  When a
    ReaderWriter has both its reader and writer parts active in the same
    step, the read and write phases are serialised (each costs at least 1
    cycle).
    """
    num_levels = len(tiling)
    if num_levels == 0:
        return 0

    bounds = [t[1] for t in tiling]
    num_ops = len(operand_descriptors)

    # Total number of iteration steps
    total_steps = 1
    for b in bounds:
        total_steps *= b

    def step_to_counters(step: int) -> list[int]:
        """Convert a global step to per-level counters (inner-first)."""
        counters = []
        s = step
        for b in bounds:
            counters.append(s % b)
            s //= b
        return counters

    def compute_bank_address(op_idx: int, counters: list[int]) -> int:
        addr = 0
        for lvl in range(num_levels):
            addr += strides_bank[op_idx][lvl] * counters[lvl]
        return addr % num_banks

    total_cycles = 0

    for step in range(total_steps):
        counters = step_to_counters(step)

        # Collect read-phase accesses (readers + RW reader parts)
        read_starts: list[int] = []
        read_widths: list[int] = []
        # Collect write-phase accesses (writers + RW writer parts)
        write_starts: list[int] = []
        write_widths: list[int] = []

        rw_stall = False

        for op_idx, desc in enumerate(operand_descriptors):
            if desc.spatial_banks <= 0:
                continue

            if desc.kind == OperandKind.READER:
                if _reader_active(counters, tiling, desc.invariant_dims):
                    addr = compute_bank_address(op_idx, counters)
                    read_starts.append(addr)
                    read_widths.append(desc.spatial_banks)

            elif desc.kind == OperandKind.WRITER:
                if _writer_active(counters, bounds, tiling, desc.invariant_dims):
                    addr = compute_bank_address(op_idx, counters)
                    write_starts.append(addr)
                    write_widths.append(desc.spatial_banks)

            elif desc.kind == OperandKind.READER_WRITER:
                # Reader part uses current step's counters
                reader_is_active = _reader_active(
                    counters, tiling, desc.invariant_dims
                )

                # Writer part is 2 global steps behind
                writer_is_active = False
                writer_counters: list[int] | None = None
                writer_step = step - 2
                if writer_step >= 0:
                    writer_counters = step_to_counters(writer_step)
                    writer_is_active = _writer_active(
                        writer_counters, bounds, tiling, desc.invariant_dims
                    )

                if reader_is_active:
                    addr = compute_bank_address(op_idx, counters)
                    read_starts.append(addr)
                    read_widths.append(desc.spatial_banks)

                if writer_is_active:
                    assert writer_counters is not None
                    addr = compute_bank_address(op_idx, writer_counters)
                    write_starts.append(addr)
                    write_widths.append(desc.spatial_banks)

                if reader_is_active and writer_is_active:
                    rw_stall = True

        if rw_stall:
            # Read and write phases are serialised for the RW operand.
            read_hits = _count_max_bank_hits(
                read_starts, read_widths, num_banks
            )
            write_hits = _count_max_bank_hits(
                write_starts, write_widths, num_banks
            )
            total_cycles += max(1, read_hits) + max(1, write_hits)
        else:
            # All accesses happen simultaneously.
            all_starts = read_starts + write_starts
            all_widths = read_widths + write_widths
            hits = _count_max_bank_hits(all_starts, all_widths, num_banks)
            total_cycles += max(hits, 1)  # always at least 1 cycle

    # Drain cycles for RW writer pipeline (2-step lag)
    for desc in operand_descriptors:
        if desc.kind == OperandKind.READER_WRITER:
            total_cycles += 2
            break  # only count once

    return total_cycles


# ---------------------------------------------------------------------------
# Convenience: differential overlap check (for external use / unit tests)
# ---------------------------------------------------------------------------

def check_burst_overlap_differential(
    stride_diff_mod: int,
    burst_width_a: int,
    burst_width_b: int,
    num_banks: int,
) -> bool:
    """
    Quick check whether two bursts that are ``stride_diff_mod`` banks apart
    (modulo ``num_banks``) overlap.

    Two contiguous bursts of widths *w_a* and *w_b* starting at banks
    *b* and *b + d* overlap iff ``d < w_a`` or ``num_banks - d < w_b``
    (wrapping case).
    """
    d = stride_diff_mod % num_banks
    if d == 0:
        return True  # exact same start
    return d < burst_width_a or (num_banks - d) < burst_width_b


# ---------------------------------------------------------------------------
# Factory / selector
# ---------------------------------------------------------------------------

def get_cost_model(name: str = "latency") -> CostModel:
    """Return a cost model instance by name."""
    models: dict[str, CostModel] = {
        "energy": EnergyCostModel(),
        "latency": LatencyCostModel(),
    }
    if name not in models:
        raise ValueError(f"Unknown cost model '{name}'. Choose from {list(models)}")
    return models[name]