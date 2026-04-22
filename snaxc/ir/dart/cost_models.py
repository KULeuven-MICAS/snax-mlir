"""
Cost models for the SNAX DART scheduler.

This module provides cost model implementations for evaluating tiling strategies:
- EnergyCostModel: Naive model minimizing total TCDM accesses (proxy for energy).
- LatencyCostModel: Step-level banking-conflict model (legacy).
- HardwareLatencyCostModel: Cycle-accurate simulation matching the RTL streamer
  pipeline (readers/writers with depth-2 buffers, burst-level banking conflicts,
  accelerator fire gating, and round-robin bank arbitration).
"""

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import reduce
from math import gcd
from operator import mul
from typing import Protocol

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Streamer / operand metadata needed by the cost models
# ---------------------------------------------------------------------------

class OperandKind(Enum):
    """Whether the operand is read, written, or both."""
    READER = auto()
    WRITER = auto()
    READER_WRITER = auto()


TCDM_BANK_BYTES = 8  # Each TCDM bank holds one 64-bit word = 8 bytes


@dataclass(frozen=True)
class OperandDescriptor:
    """
    Lightweight description of a single operand's access behaviour that is
    independent of any particular tiling.

    Attributes:
        kind: READER, WRITER, or READER_WRITER.
        spatial_banks: Number of spatial elements accessed per burst.
            Equal to (product of spatial dims).
        element_bytes: Size of a single element in bytes (e.g. 1 for i8,
            4 for i32).  Needed to convert element counts to byte/bank
            addresses.
        invariant_dims: Set of *template* dimension indices to which the
            operand is invariant (i.e. the stride is 0 for those dims).
            Callers must convert from logical dimension IDs (L_IDs) to
            template dimension indices before constructing descriptors.
    """
    kind: OperandKind
    spatial_banks: int
    element_bytes: int
    invariant_dims: frozenset[int]
    fixed_cache_depth: int = 64

    @property
    def burst_bank_words(self) -> int:
        """Number of TCDM bank words touched per spatial burst.

        Each bank word is ``TCDM_BANK_BYTES`` (8) bytes.  A burst
        accesses ``spatial_banks * element_bytes`` contiguous bytes,
        which spans ``spatial_banks * element_bytes / TCDM_BANK_BYTES``
        consecutive banks.
        """
        return self.spatial_banks


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
        template_bounds: tuple[int, ...] | None = None,
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
        template_bounds: tuple[int, ...] | None = None,
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
        template_bounds: tuple[int, ...] | None = None,
    ) -> float:
        return latency_cost_of_tiling(
            tiling,
            operand_descriptors,
            invariance_map,
            num_banks=num_banks,
            bank_bits=bank_bits,
            template_bounds=template_bounds,
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
    *,
    template_bounds: tuple[int, ...] | None = None,
    data_shapes: Sequence[Sequence[int]] | None = None,
) -> list[list[int]]:
    """
    For every operand, compute the *byte stride* contributed by each tiling
    level.

    Replicates the compiler pipeline (``rebuild_schedule`` →
    ``set_memory_layout`` → ``dart_layout_resolution``) to derive the
    strides that the hardware streamers would actually use.

    Parameters
    ----------
    template_bounds : per-template-dimension spatial tile sizes
        (e.g. ``(8, 8, 8)`` for GEMMX).  When provided the TSL-based
        stride derivation is used; otherwise falls back to the legacy
        ``burst_bank_words * cum`` formula.
    data_shapes : per-operand memref shapes, one tuple per operand, each
        with ``num_data_dims`` entries.  When *None* the shapes are
        inferred from *tiling* and *template_bounds*.

    Returns
    -------
    strides : list[list[int]]
        ``strides[op_idx][level_idx]`` – byte stride for that operand at
        that tiling level.
    """
    num_ops = len(operand_descriptors)

    # ------------------------------------------------------------------
    # Fast path: legacy formula when template_bounds is not provided
    # ------------------------------------------------------------------
    if template_bounds is None:
        dim_cumulative: dict[int, int] = {}
        strides: list[list[int]] = [[] for _ in range(num_ops)]
        for _lvl, (dim_idx, tile_size, _is_crit) in enumerate(tiling):
            cum = dim_cumulative.get(dim_idx, 1)
            for op_idx, desc in enumerate(operand_descriptors):
                if dim_idx in desc.invariant_dims:
                    strides[op_idx].append(0)
                else:
                    strides[op_idx].append(desc.burst_bank_words * cum)
            dim_cumulative[dim_idx] = cum * tile_size
        return strides

    # ------------------------------------------------------------------
    # TSL-based stride derivation (replicates compiler passes)
    # ------------------------------------------------------------------
    num_template_dims = len(template_bounds)

    # Filter size-1 levels for the schedule pattern construction
    tiling_filtered = [(d, s) for d, s, *_ in tiling if s > 1]

    # --- Per-operand access mapping ---
    # From invariant_dims, derive which template dims each operand accesses
    all_dims = set(range(num_template_dims))
    op_accessed_dims: list[list[int]] = []
    for desc in operand_descriptors:
        accessed = sorted(all_dims - set(desc.invariant_dims))
        op_accessed_dims.append(accessed)

    num_data_dims = len(op_accessed_dims[0])

    # --- Base vectors and spatial columns ---
    # base_vecs[template_dim][op_idx] = array of len(num_data_dims)
    base_vecs: dict[int, list[np.ndarray]] = {}
    spatial_cols: list[list[np.ndarray]] = []

    for tdim in range(num_template_dims):
        bvecs = []
        svecs = []
        for op_idx, accessed in enumerate(op_accessed_dims):
            bv = np.zeros(num_data_dims, dtype=int)
            sv = np.zeros(num_data_dims, dtype=int)
            if tdim in accessed:
                data_dim_idx = accessed.index(tdim)
                bv[data_dim_idx] = template_bounds[tdim]
                sv[data_dim_idx] = 1
            bvecs.append(bv)
            svecs.append(sv)
        base_vecs[tdim] = bvecs
        spatial_cols.append(svecs)

    # --- Build schedule pattern A (outer→inner temporal + spatial) ---
    stride_tracker: dict[int, int] = {d: 1 for d, _ in tiling_filtered}
    for d in range(num_template_dims):
        stride_tracker.setdefault(d, 1)

    components: list[tuple[int, list[np.ndarray]]] = []
    for orig_idx, size in tiling_filtered:
        vecs = [base_vecs[orig_idx][op] * stride_tracker[orig_idx]
                for op in range(num_ops)]
        components.append((size, vecs))
        stride_tracker[orig_idx] *= size

    components.reverse()  # outer → inner

    bounds = [x[0] for x in components] + list(template_bounds)

    patterns: list[np.ndarray] = []
    for op_idx in range(num_ops):
        temp_cols = [x[1][op_idx] for x in components]
        spat = [spatial_cols[d][op_idx] for d in range(num_template_dims)]
        cols = temp_cols + spat
        patterns.append(np.column_stack(cols) if cols
                        else np.zeros((num_data_dims, 0), dtype=int))

    # --- Infer data shapes if not provided ---
    if data_shapes is None:
        dim_total: dict[int, int] = {d: template_bounds[d]
                                     for d in range(num_template_dims)}
        for d, s, *_ in tiling:
            if s > 1:
                dim_total[d] *= s
        data_shapes = []
        for accessed in op_accessed_dims:
            data_shapes.append(tuple(dim_total[d] for d in accessed))

    # --- Simulate set_memory_layout for each operand → TSL ---
    def _simulate_tsl(
        pattern: np.ndarray,
        bnds: list[int],
        shape: Sequence[int],
        elem_b: int,
    ) -> list[list[tuple[int, int]]]:
        """Return TSL per data dim as [(stride_bytes, bound)] inner→outer."""
        nd = pattern.shape[0]
        current_stride = 1  # in elements
        raw: list[list[tuple[int, int]]] = [[] for _ in range(nd)]

        for rev_idx in range(len(bnds)):
            orig_idx = len(bnds) - 1 - rev_idx
            sbound = bnds[orig_idx]
            accesses = pattern[:, orig_idx]
            abin = tuple(0 if x == 0 else 1 for x in accesses)

            if 1 not in abin:
                continue

            # ensure_access_granularity
            if current_stride != 1:
                if rev_idx >= num_template_dims:  # temporal
                    gran = 8 if elem_b == 1 else 16
                else:  # spatial
                    gran = 8 if elem_b == 1 else 2
                if current_stride % gran != 0:
                    current_stride += (gran - current_stride) % 64

            adim = abin.index(1)
            existing_bound = 1
            for s, b in raw[adim]:
                existing_bound *= b
            size_remaining = shape[adim] // existing_bound

            to_tile = True
            if size_remaining % sbound != 0:
                to_tile = False
            else:
                for sv, bv in zip(pattern[adim, :], bnds):
                    if sv % sbound != 0 and bv != sbound:
                        to_tile = False
                        break

            lbound = sbound if to_tile else size_remaining
            raw[adim].insert(0, (current_stride, lbound))
            current_stride *= lbound

        for dim_strides in raw:
            if not dim_strides:
                dim_strides.append((current_stride, 1))

        # Convert to bytes, reverse to inner→outer, remove bound-1 entries
        tsl: list[list[tuple[int, int]]] = []
        for dim_strides in raw:
            inner_first = [(s * elem_b, b) for s, b in reversed(dim_strides)]
            inner_first = [(s, b) for s, b in inner_first if b != 1]
            if not inner_first:
                inner_first = [(0, 1)]
            tsl.append(inner_first)
        return tsl

    def _eval_tsl(entries: list[tuple[int, int]], x: int) -> int:
        result = 0
        for stride, bound in entries:
            result += (x % bound) * stride
            x //= bound
        return result

    # --- Compose TSL with pattern to get per-tiling-level byte strides ---
    num_temporal = len(tiling_filtered)

    result_strides: list[list[int]] = [[] for _ in range(num_ops)]
    for op_idx in range(num_ops):
        pat = patterns[op_idx]
        tsl = _simulate_tsl(pat, bounds, data_shapes[op_idx],
                            operand_descriptors[op_idx].element_bytes)

        # Pre-compute byte stride for each schedule dim
        sched_strides: list[int] = []
        for col_idx in range(len(bounds)):
            col = pat[:, col_idx]
            addr = sum(_eval_tsl(tsl[d], int(col[d]))
                       for d in range(num_data_dims))
            sched_strides.append(addr)

        # Map schedule dims → tiling levels (inner→outer)
        fi = 0
        for _dim, size, *_ in tiling:
            if size == 1:
                result_strides[op_idx].append(0)
            else:
                sched_col = num_temporal - 1 - fi
                result_strides[op_idx].append(sched_strides[sched_col])
                fi += 1

    return result_strides


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
    template_bounds: tuple[int, ...] | None = None,
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
    template_bounds : per-template-dimension spatial tile sizes.

    Returns
    -------
    float  – total cycles.
    """
    if not tiling:
        return 0.0

    # Compute per-operand strides in bank-word units
    strides_bank = _compute_operand_stride_per_tile(
        tiling, operand_descriptors, invariance_map, bank_bits,
        template_bounds=template_bounds,
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

    cycles_for_step_i = []
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
            bw = desc.burst_bank_words
            if bw <= 0:
                continue

            if desc.kind == OperandKind.READER:
                if _reader_active(counters, tiling, desc.invariant_dims):
                    addr = compute_bank_address(op_idx, counters)
                    read_starts.append(addr)
                    read_widths.append(bw)

            elif desc.kind == OperandKind.WRITER:
                if _writer_active(counters, bounds, tiling, desc.invariant_dims):
                    addr = compute_bank_address(op_idx, counters)
                    write_starts.append(addr)
                    write_widths.append(bw)

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
                    read_widths.append(bw)

                if writer_is_active:
                    assert writer_counters is not None
                    addr = compute_bank_address(op_idx, writer_counters)
                    write_starts.append(addr)
                    write_widths.append(bw)

                if reader_is_active and writer_is_active:
                    rw_stall = True
        if step == 15:
            pass
        if rw_stall:
            # Read and write phases are serialised for the RW operand.
            read_hits = _count_max_bank_hits(
                read_starts, read_widths, num_banks
            )
            write_hits = _count_max_bank_hits(
                write_starts, write_widths, num_banks
            )
            total_cycles += max(1, read_hits) + max(1, write_hits)
            cycles_for_step_i.append(max(1, read_hits) + max(1, write_hits))
        else:
            # All accesses happen simultaneously.
            all_starts = read_starts + write_starts
            all_widths = read_widths + write_widths
            hits = _count_max_bank_hits(all_starts, all_widths, num_banks)
            total_cycles += max(hits, 1)  # always at least 1 cycle
            cycles_for_step_i.append(max(hits, 1))

    # Drain cycles for RW writer pipeline (2-step lag)
    for desc in operand_descriptors:
        if desc.kind == OperandKind.READER_WRITER:
            total_cycles += 2
            break  # only count once

    # plt.plot(cycles_for_step_i)
    # plt.xlabel("Global iteration step")
    # plt.ylabel("Cycles")
    # plt.title("Cycles per iteration step")
    # plt.savefig("cycles_per_step.png")

    return total_cycles


# ---------------------------------------------------------------------------
# Hardware-accurate latency cost model – cycle-level RTL simulation
# ---------------------------------------------------------------------------

class HardwareLatencyCostModel(CostModel):
    """
    Cycle-accurate cost model that simulates the actual SNAX streamer
    hardware pipeline.

    Each streamer has:
      - Its own step counter (incremented when a memory access completes or
        when the accelerator fires for a non-access step).
      - A buffer of depth 2 holding step indices.
      - Burst accesses of ``spatial_banks`` individual bank words; partial
        completion is tracked per-bank so banking conflicts only stall the
        conflicting sub-accesses.

    The accelerator fires when:
      - Every reader-type streamer either has the required data in its buffer
        *or* does not need a memory access at the current accelerator step.
      - Every writer-type streamer has space in its buffer.

    Banking conflicts are resolved with round-robin priority among all
    requestors competing for the same bank in a given cycle.
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
        template_bounds: tuple[int, ...] | None = None,
    ) -> float:
        return hardware_latency_cost_of_tiling(
            tiling,
            operand_descriptors,
            invariance_map,
            num_banks=num_banks,
            bank_bits=bank_bits,
            sparse_port_config=GEMMX_SPARSE_PORT_CONFIG,  # type: ignore[arg-type]
            dormant_ports=GEMMX_DORMANT_PORTS,
            template_bounds=template_bounds,
        )


def hardware_latency_cost_of_tiling(
    tiling: list[tuple[int, int, bool]],
    operand_descriptors: Sequence[OperandDescriptor],
    invariance_map: list[set[int]],
    *,
    num_banks: int = 32,
    bank_bits: int = 64,
    sparse_port_config: "list[SparsePortDef] | None" = None,
    dormant_ports: frozenset[int] | None = None,
    template_bounds: tuple[int, ...] | None = None,
) -> float:
    """
    Cycle-accurate simulation of the SNAX streamer hardware.

    Returns the total number of TCDM clock cycles to complete all streamer
    accesses.
    """
    if not tiling:
        return 0.0

    strides_bank = _compute_operand_stride_per_tile(
        tiling, operand_descriptors, invariance_map, bank_bits,
        template_bounds=template_bounds,
    )

    return float(_simulate_hardware(
        tiling, operand_descriptors, invariance_map, strides_bank, num_banks,
        sparse_port_config=sparse_port_config,
        dormant_ports=dormant_ports,
    ))


# ---------------------------------------------------------------------------
# Internal: hardware simulation — per-streamer state classes
# ---------------------------------------------------------------------------


@dataclass
class AGUState:
    """
    Models the AddressGenUnit FSM and counter chain from the RTL.

    The AGU has a chain of ProgrammableCounters (one per tiling level).
    On each ``counters_tick`` the innermost counter increments; when it
    wraps the next counter increments, etc.  A ``counters_tick`` fires
    whenever the outputBuffer (and, when caching, the
    fixedCacheInstructionBuffer) successfully accepts a new entry.

    The outputBuffer is a FIFO of depth ``output_buffer_depth`` that
    stores per-channel bank addresses (one entry = addresses for all
    spatial channels of this streamer).
    """

    # --- Configuration (set once) ---
    num_levels: int
    bounds: list[int]  # tile sizes per level (inner→outer)
    is_writer: bool  # True for WRITER kind
    output_buffer_depth: int = 4

    # --- Runtime state ---
    counters: list[int] = field(default_factory=list)
    state: str = "IDLE"  # "IDLE" or "BUSY"
    done: bool = False  # all counters wrapped → AGU finished

    def __post_init__(self):
        if not self.counters:
            self.counters = [0] * self.num_levels

    def reset_and_start(self):
        """Transition IDLE → BUSY, zero all counters."""
        self.counters = [0] * self.num_levels
        self.state = "BUSY"
        self.done = False

    def current_step(self) -> int:
        """Convert current counter state to a global step index."""
        step = 0
        multiplier = 1
        for lvl in range(self.num_levels):
            step += self.counters[lvl] * multiplier
            multiplier *= self.bounds[lvl]
        return step

    def tick(self) -> bool:
        """
        Advance the counter chain by one tick (innermost first, carry
        propagation).  Returns True if the outermost counter just
        wrapped (i.e. this was the very last tick of the whole
        sequence).
        """
        for lvl in range(self.num_levels):
            self.counters[lvl] += 1
            if self.counters[lvl] < self.bounds[lvl]:
                return False
            self.counters[lvl] = 0  # wrap, carry to next level
        # All counters wrapped → done
        self.done = True
        self.state = "IDLE"
        return True


@dataclass
class StreamerState:
    """
    Models the full state of a single hardware streamer (Reader or
    Writer), mirroring the RTL pipeline:

        AGU → outputBuffer (per-channel addr FIFOs)
            → DataRequestors (one per channel, each issues to TCDM)
            → DataResponsers (reader) / TCDM ack (writer)
            → dataBuffer (depth-2 data FIFO)
            → accelerator interface

    The output buffer is implemented as ``spatial_banks`` independent
    FIFOs (one per channel), matching the hardware's
    ``ComplexQueueConcat`` which instantiates one ``Queue`` per
    channel.  All channels enqueue simultaneously when the AGU ticks
    (requiring all to have room), but each channel dequeues
    independently when its DataRequestor is granted access to its
    TCDM bank.

    Without fixed-cache support (``newUseCache = false``), the AGU
    places an address into the output buffer on *every* counter tick.
    ``counters_tick`` fires iff ``currentState == sBUSY &&
    outputBuffer.io.in.head.fire`` – so the counter only advances
    when the buffer actually accepts.  No steps are skipped.
    """

    # --- Configuration (set once) ---
    op_idx: int
    kind: OperandKind  # READER or WRITER (RW split into two)
    spatial_banks: int
    invariant_dims: frozenset[int]
    num_levels: int
    bounds: list[int]
    strides_bank: list[int]  # per-level strides in bank-word units
    num_banks: int
    tiling: list[tuple[int, int, bool]]

    output_buffer_depth: int = 4
    data_buffer_depth: int = 2
    fixed_cache_depth: int = 64

    # True for streamers created from a READER_WRITER split.
    from_rw: bool = False

    # Index of the paired writer in the streamers list (set for
    # from_rw reader streamers).  Used by the prediction logic to
    # find the writerPort state for dual-port bank conflict checks.
    _paired_writer_idx: int | None = field(default=None, repr=False)

    # HandShakeRepeater count — models the Reader's RepeatHandshake.
    # When the innermost loop has stride 0 and bound > 1, the reader
    # overrides AGU bounds[0] to 1 and repeats each data word
    # ``repeat_count`` times.  Default 1 = no repeat.
    repeat_count: int = 1

    # Level whose bound was overridden to 1 by the repeat override.
    # None if no override was applied.  Used to avoid pruning this
    # level in uses_fixed_cache.
    _repeat_override_level: int | None = field(default=None, repr=False)

    # --- Sub-components ---
    agu: AGUState = field(default=None)  # type: ignore[assignment]

    # Per-channel output buffer queues (N independent FIFOs,
    # matching ComplexQueueConcat = N independent Queue modules).
    # output_buffers[ch] is a deque of step indices for channel ch.
    output_buffers: list[deque[int]] = field(default_factory=list)

    # Per-channel pending bank request.  Each entry is the bank index
    # that channel ``i`` is currently requesting, or None if channel
    # ``i`` is idle (not requesting).
    channel_pending_bank: list[int | None] = field(default_factory=list)

    # Track step completion: step → number of channels still to grant.
    # Initialised to spatial_banks when the AGU pushes a step.
    # Decremented on each channel grant.  When it reaches 0 the step
    # is fully complete and its response can enter the pipeline.
    step_grants_remaining: dict[int, int] = field(default_factory=dict)

    # dataBuffer: FIFO of step indices (reader: fetched data from TCDM;
    # for writers this is only used in the fixed-cache path).
    data_buffer: deque[int] = field(default_factory=deque)

    # Writer per-channel data buffers.  Models the ComplexQueueConcat
    # that splits one wide accelerator output into N per-channel queues
    # (each of depth data_buffer_depth).  Each channel independently
    # dequeues when its TCDM request fires (grant).  The wide input
    # is accepted only when ALL per-channel queues have room.
    writer_channel_bufs: list[deque[int]] = field(default_factory=list)

    # For readers: per-channel response slots available.  Each channel
    # has its own UpDownCounter in DataResponser with ceil = bufferDepth+1.
    # tickUp fires per-channel on reqSubmit; tickDown fires for ALL
    # channels simultaneously on dataBuffer.out.fire (dataFifoPopped).
    responser_slots: list[int] = field(default_factory=list)

    # ---- FixedLevelCache state (only active when uses_fixed_cache) ----
    # Writer pipe=false: the writer's ComplexQueueConcat has pipe=false,
    # meaning the buffer cannot accept new data in the same cycle it
    # becomes non-full due to a dequeue.  This flag snapshots whether
    # the writer buffer was full at the start of a cycle (before
    # Phase 0 pops), so that writer_has_space() correctly returns
    # False for the rest of that cycle even after the pop.
    _writer_buf_full_at_start: bool = False

    # --- Writer cache state (old model, used for WRITER streamers) ---
    # Models the fixedCacheInstructionBuffer FIFO in the Writer AGU.
    # Each entry is ``(step, is_update_cache, cache_index)`` where
    # ``is_update_cache`` means the step needs TCDM data and
    # ``not is_update_cache`` means data is read from cache memory.
    # ``cache_index`` is the computed FixedLevelCache index for this
    # step (used by the writerPort to write into the reader's cache).
    # FIFO depth = output_buffer_depth.
    cache_instruction_buffer: deque[tuple[int, bool, int]] = field(default_factory=deque)
    old_cib: deque[tuple[int, bool, int]] = field(default_factory=deque)

    # When the writer fires with goToCache (useCache && !lastAccess),
    # this records the cache index the writerPort writes to.  Set in
    # writer_accept, cleared at the start of each cycle.  The paired
    # reader's process_reader_cache uses this to determine writerPort
    # bank conflicts on the dual-port SRAM write port.
    _writer_port_cache_index: int | None = None

    # Models the ``delayedValid`` register + associated data inside
    # the FixedLevelCache.  When not None, the cache has valid output
    # for this step that the accelerator can consume.  Updated at the
    # end of each cycle (register semantics).
    cache_output_step: int | None = None

    # --- Reader cache state (new model, separate write/read buffers) ---
    # Models the new AddressGenUnitReader with three independent counter
    # chains and the dual-bank FixedLevelCache.

    # Write cache instruction buffer (stores cache indices).
    # Consumed by FixedLevelCache together with data from data_buffer.
    write_cache_buffer: deque[int] = field(default_factory=deque)

    # Read cache instruction buffer (stores (cache_index, agu_step, last_access) tuples).
    # Consumed by FixedLevelCache to produce output to accelerator.
    read_cache_buffer: deque[tuple[int, int, bool]] = field(default_factory=deque)

    # Independent write cache counter positions (per-level, 0..bounds[i]-1)
    write_cache_positions: list[int] = field(default_factory=list)
    write_cache_done: bool = False

    # Independent read cache counter positions
    read_cache_positions: list[int] = field(default_factory=list)
    read_cache_done: bool = False

    # Derived cache parameters (computed in __post_init__)
    _critical_loop_level: int = field(default=-1, repr=False)
    _total_bounds_arr: list[int] = field(default_factory=list, repr=False)
    _fixed_cache_period: int = field(default=1, repr=False)

    # Original bounds for read cache counters (before ceil=1 modification).
    # Only populated for readers with fixed cache.
    _read_cache_bounds: list[int] = field(default_factory=list, repr=False)
    # Cached result of uses_fixed_cache (computed before bounds modification).
    _uses_fixed_cache: bool | None = field(default=None, repr=False)

    # Period/servicing tracking (matches RTL registers)
    write_period_count: int = 0
    read_serviced_period_count: int = 0
    serviced_counter: int = 0  # write instructions consumed by FixedLevelCache
    read_issued_counter: int = 0  # read counter ticks at updateCache positions

    # FixedLevelCache dual-bank pipeline state (registered)
    cache_instr_valid: bool = False
    cache_instr_step: int = -1  # AGU step for current read instruction
    cache_instr_index: int = 0  # cache index for current read instruction
    cache_data_held: bool = False
    cache_read_data_arriving: bool = False

    def __post_init__(self):
        # --- Repeat / bound override for ReaderWriter streamers ---
        # Bound-1 levels from the original configuration are pruned
        # (treated as no-ops).  The effective innermost level is the
        # first with bound > 1.
        #
        # For ReaderWriter (from_rw) streamers only:
        #   Reader.scala: when(temporalStrides(0) === 0.U) {
        #     addressgen.io.cfg.temporalBounds(0) := 1.U
        #   }
        #   HandShakeRepeater: repeat_times = Mux(stride(0)==0, bounds(0), 1)
        #   Both the reader and writer halves of the RW pair share the
        #   same AGU configuration, so both see the bound override.
        #   The new bound-1 must NOT be pruned later.
        if self.from_rw:
            effective_inner = None
            for lvl in range(len(self.strides_bank)):
                if self.bounds[lvl] > 1:
                    effective_inner = lvl
                    break
            if (effective_inner is not None
                    and self.strides_bank[effective_inner] == 0):
                self.bounds = list(self.bounds)  # ensure own copy
                self.bounds[effective_inner] = 1
                self._repeat_override_level = effective_inner
        else:
            effective_inner = None
            for lvl in range(len(self.strides_bank)):
                if self.bounds[lvl] > 1:
                    effective_inner = lvl
                    break
            if (effective_inner is not None
                    and self.strides_bank[effective_inner] == 0):
                self.repeat_count = self.bounds[effective_inner]
                self.bounds = list(self.bounds)  # ensure own copy
                self.bounds[effective_inner] = 1

        if self.agu is None:
            self.agu = AGUState(
                num_levels=self.num_levels,
                bounds=self.bounds,
                is_writer=(self.kind == OperandKind.WRITER),
                output_buffer_depth=self.output_buffer_depth,
            )
        if not self.output_buffers:
            self.output_buffers = [deque() for _ in range(self.spatial_banks)]
        if not self.channel_pending_bank:
            self.channel_pending_bank = [None] * self.spatial_banks
        if not self.responser_slots:
            # Each channel starts with data_buffer_depth free slots
            self.responser_slots = [self.data_buffer_depth] * self.spatial_banks
        if not self.writer_channel_bufs and self.kind == OperandKind.WRITER:
            self.writer_channel_bufs = [deque() for _ in range(self.spatial_banks)]

        # --- Initialise cache state ---
        self._uses_fixed_cache = self.uses_fixed_cache
        if self._uses_fixed_cache:
            self._init_cache_params()
        if self.is_reader and self._uses_fixed_cache:
            self._init_reader_cache()
            # Apply ceil=1 for irrelevant dims in reader AGU and
            # write cache counters (matching RTL writeCounterCeil).
            # Read cache counters keep original bounds.
            self._read_cache_bounds = list(self.bounds)
            cl = self._critical_loop_level
            self.bounds = list(self.bounds)
            for i in range(self.num_levels):
                if self.strides_bank[i] == 0 and i <= cl:
                    self.bounds[i] = 1
            self.agu.bounds = list(self.bounds)

    def _init_cache_params(self):
        """Compute totalBounds and criticalLoopLevel for cache index
        computation.  Needed by both reader and writer streamers."""
        bounds = self.bounds
        strides = self.strides_bank
        n = self.num_levels
        total_bounds = [0] * n
        total_bounds[0] = bounds[0]
        for i in range(1, n):
            if strides[i] == 0:
                total_bounds[i] = total_bounds[i - 1]
            else:
                total_bounds[i] = total_bounds[i - 1] * bounds[i]
        self._total_bounds_arr = total_bounds
        critical_loop = -1
        for i in range(n - 1, 0, -1):
            if (total_bounds[i - 1] <= self.fixed_cache_depth
                    and strides[i] == 0
                    and bounds[i] > 1):
                critical_loop = i
                break
        self._critical_loop_level = critical_loop

    def _init_reader_cache(self):
        """Initialise reader cache state (totalBounds and criticalLoop
        already computed by _init_cache_params)."""
        n = self.num_levels
        critical_loop = self._critical_loop_level
        total_bounds = self._total_bounds_arr

        self._fixed_cache_period = (
            total_bounds[critical_loop - 1] if critical_loop > 0 else 1
        )

        # Initialise counter positions
        if not self.write_cache_positions:
            self.write_cache_positions = [0] * n
        if not self.read_cache_positions:
            self.read_cache_positions = [0] * n

    # ----- Reader cache counter helpers -----

    def _is_update_cache_at(self, positions: list[int]) -> bool:
        """Check if all invariant loops within cache scope are at zero.

        Matches the RTL updateCacheConditions logic:
        For every dimension i where stride==0 AND i <= criticalLoop,
        the counter at that dimension must be 0 (isZero).
        """
        cl = self._critical_loop_level
        for i in range(self.num_levels):
            if self.strides_bank[i] == 0 and i <= cl:
                if positions[i] != 0:
                    return False
        return True

    def _is_read_last_access_at(self, positions: list[int]) -> bool:
        """Check if all irrelevant counters within cache scope are at
        their last value (matching RTL ReadLastAccessConditions).

        For each dimension where stride==0 AND i <= criticalLoop,
        the read counter must be at its last value (bound-1 in
        _read_cache_bounds, which holds the original unmodified bounds).
        """
        cl = self._critical_loop_level
        for i in range(self.num_levels):
            if self.strides_bank[i] == 0 and i <= cl:
                if positions[i] != self._read_cache_bounds[i] - 1:
                    return False
        return True

    def _compute_cache_index_at(self, positions: list[int]) -> int:
        """Compute the cache index from counter positions.

        Matches the RTL writeIndex / readIndex computation:
        sum of position[i] * step_size[i] for non-invariant levels
        within cache scope, where step_size[0]=1 and
        step_size[i]=totalBounds[i-1].
        """
        cl = self._critical_loop_level
        tb = self._total_bounds_arr
        index = 0
        for i in range(self.num_levels):
            if self.strides_bank[i] != 0 and i <= cl:
                step_size = 1 if i == 0 else tb[i - 1]
                index += positions[i] * step_size
        return index

    def _cache_counter_tick(self, positions: list[int], bounds: list[int] | None = None) -> bool:
        """Advance a cache counter chain by one tick.

        Models the ProgrammableCounter cascading in the RTL:
        level 0 always ticks, level i>0 ticks when level i-1
        was at lastVal (position == bounds-1) before the tick.

        Args:
            bounds: explicit bounds to use.  Defaults to self.bounds
                (modified bounds with ceil=1 for irrelevant dims).
                Pass self._read_cache_bounds for the read cache
                counter which uses the original unmodified bounds.

        Returns True if ALL levels were at lastVal (sequence done).
        """
        bnds = bounds if bounds is not None else self.bounds
        # Snapshot pre-tick lastVals
        pre_at_last = [
            positions[lvl] == bnds[lvl] - 1
            for lvl in range(self.num_levels)
        ]
        all_done = True
        for lvl in range(self.num_levels):
            if lvl > 0 and not pre_at_last[lvl - 1]:
                all_done = False
                break
            # Tick this level
            positions[lvl] += 1
            if positions[lvl] >= bnds[lvl]:
                positions[lvl] = 0
            if not pre_at_last[lvl]:
                all_done = False
                break
        return all_done

    def _cache_data_available(self) -> bool:
        """FixedLevelCache output data is available (combinational)."""
        return self.cache_instr_valid and (
            self.cache_data_held or self.cache_read_data_arriving
        )

    def process_reader_cache(self, acc_consumed: bool, writer_port_cache_index: int | None = None):
        """Process one cycle of the reader's FixedLevelCache and cache
        counter ticks.  Called AFTER Phase 4 (acc tick) with the known
        acc consumption result.

        For ReaderWriter (from_rw) streamers, dual-port (1W1R) SRAMs
        are used: the read port is independent of the write port, so
        reads never conflict with writes.  However, the writerPort
        (driven by the writer half) shares the write port with TCDM
        writes, so TCDM writes stall when the writerPort targets the
        same bank.

        For regular readers (not from_rw), single-port SRAMs are used:
        reads have priority and block writes to the same bank.

        Args:
            acc_consumed: whether the accelerator consumed the reader's
                output this cycle.
            writer_port_cache_index: if not None, the cache index the
                paired writer's writerPort is writing to this cycle
                (only relevant for from_rw readers).

        This method models:
        1. FixedLevelCache read/write accept decisions
        2. Buffer dequeues (from cache consumption)
        3. Cache counter ticks (write and read, with buffer readiness)
        4. Buffer enqueues (from counter ticks)
        5. Registered state updates (cache pipeline, period counters)
        """
        # ── Step 1: FixedLevelCache combinational signals ──
        data_available = self._cache_data_available()
        delivering = data_available and acc_consumed

        can_accept_new = (not self.cache_instr_valid) or delivering
        accept_new = bool(self.read_cache_buffer) and can_accept_new

        # Determine read bank if accepting
        read_bank = -1
        if accept_new:
            read_index, _step, _la = self.read_cache_buffer[0]
            read_bank = read_index & 1

        issue_read = accept_new

        # Bank conflict logic depends on memory type:
        #   from_rw (dual-port 1W1R): write blocked by writerPort on same bank
        #   regular (single-port):    write blocked by read on same bank
        write_bank = -1
        can_accept_write = False
        if self.write_cache_buffer and self.data_buffer:
            write_index = self.write_cache_buffer[0]
            write_bank = write_index & 1
            if self.from_rw:
                # Dual-port SRAM: read and write ports are independent.
                # The writerPort (from the RW writer) shares the write
                # port with TCDM writes → conflict when same bank.
                if writer_port_cache_index is not None:
                    writer_port_bank = writer_port_cache_index & 1
                    write_bank_busy = (write_bank == writer_port_bank)
                else:
                    write_bank_busy = False
                # Also stall when a read targets the same bank AND
                # same address as the TCDM write (dual-port same-
                # address read/write hazard).
                if issue_read and read_bank == write_bank:
                    write_addr = write_index >> 1
                    read_addr = read_index >> 1
                    if write_addr == read_addr:
                        write_bank_busy = True
            else:
                # Single-port SRAM: read has priority, blocks write
                # to the same bank.
                write_bank_busy = issue_read and (write_bank == read_bank)
            can_accept_write = not write_bank_busy

        write_fire = can_accept_write
        read_fire = accept_new
        data_fifo_popped = write_fire and bool(self.data_buffer)

        # ── Step 2: Execute dequeues ──
        if write_fire:
            self.write_cache_buffer.popleft()
            self.data_buffer.popleft()
            # dataFifoPopped: free responser slots for ALL channels
            for ch in range(self.spatial_banks):
                self.responser_slots[ch] += 1
            self.serviced_counter += 1

        new_instr_step = self.cache_instr_step
        new_instr_index = self.cache_instr_index
        if read_fire:
            _idx, step, last_access = self.read_cache_buffer.popleft()
            new_instr_step = step
            new_instr_index = _idx
            if last_access:
                self.read_serviced_period_count += 1

        # ── Step 3: Compute buffer readiness (after dequeues,
        #    accounting for pipe=true: room exists if not full OR
        #    if a dequeue just happened) ──
        write_buf_room = (
            len(self.write_cache_buffer) < self.output_buffer_depth
        )
        read_buf_room = (
            len(self.read_cache_buffer) < self.output_buffer_depth
        )

        # ── Step 4: Compute effective counters (combinational bypass) ──
        effective_serviced = self.serviced_counter
        # (write_fire already incremented serviced_counter above)
        effective_read_serviced_period = self.read_serviced_period_count
        # (read_fire already incremented above)

        # ── Step 5: Write cache counter tick ──
        write_is_update = self._is_update_cache_at(self.write_cache_positions)
        write_not_too_far_ahead = (
            self.write_period_count
            < effective_read_serviced_period + self._fixed_cache_period
        )
        if write_is_update:
            write_buffer_can_accept = write_buf_room and write_not_too_far_ahead
        else:
            write_buffer_can_accept = write_not_too_far_ahead

        # In the RTL, currentState stays sBUSY until ALL counter
        # chains (main + write cache + read cache) are done.  In
        # Python the main AGU sets state="IDLE" when its counters
        # wrap, so we use agu.done as a secondary "was started" flag.
        agu_active = (
            self.agu.state == "BUSY" or self.agu.done
        )

        write_tick = (
            agu_active
            and not self.write_cache_done
            and write_buffer_can_accept
        )

        if write_tick:
            # Compute index and push to buffer (only for updateCache)
            if write_is_update:
                w_idx = self._compute_cache_index_at(self.write_cache_positions)
                self.write_cache_buffer.append(w_idx)
            # Advance write counter
            if self._cache_counter_tick(self.write_cache_positions):
                self.write_cache_done = True
            self.write_period_count += 1

        # ── Step 6: Read cache counter tick ──
        read_is_update = self._is_update_cache_at(self.read_cache_positions)
        read_can_tick = read_buf_room and (
            not read_is_update
            or self.read_issued_counter < effective_serviced
        )

        read_tick = (
            agu_active
            and not self.read_cache_done
            and read_can_tick
        )

        if read_tick:
            # Compute index and AGU step, push to buffer
            r_idx = self._compute_cache_index_at(self.read_cache_positions)
            # The AGU step for this read instruction: the read cache
            # counter visits positions in the same order as the main
            # counter.  Convert positions to a flat step using the
            # original (unmodified) read cache bounds.
            r_step = 0
            multiplier = 1
            for lvl in range(self.num_levels):
                r_step += self.read_cache_positions[lvl] * multiplier
                multiplier *= self._read_cache_bounds[lvl]
            r_last_access = self._is_read_last_access_at(self.read_cache_positions)
            self.read_cache_buffer.append((r_idx, r_step, r_last_access))
            # Update read_issued_counter if at updateCache position
            if read_is_update:
                self.read_issued_counter += 1
            # Advance read counter (uses original bounds)
            if self._cache_counter_tick(self.read_cache_positions, bounds=self._read_cache_bounds):
                self.read_cache_done = True

        # ── Step 7: Update FixedLevelCache registered state ──
        # instrValid, dataHeld, readDataArriving update
        if accept_new:
            self.cache_instr_valid = True
            self.cache_instr_step = new_instr_step
            self.cache_instr_index = new_instr_index
            self.cache_data_held = False
        elif delivering and not accept_new:
            self.cache_instr_valid = False

        new_read_data_arriving = issue_read

        if accept_new:
            self.cache_data_held = False
        elif self.cache_read_data_arriving:
            self.cache_data_held = True
        elif delivering:
            self.cache_data_held = False

        self.cache_read_data_arriving = new_read_data_arriving

        # Update cache_output_step for accelerator interface
        if self.cache_instr_valid and (
            self.cache_data_held or self.cache_read_data_arriving
        ):
            self.cache_output_step = self.cache_instr_step
        else:
            self.cache_output_step = None

    # ----- address helpers -----

    def compute_channel_bank(self, step: int, ch: int) -> int:
        """Compute the TCDM bank index for channel ``ch`` at ``step``.

        Mirrors the RTL address computation:
          addr = temporal_base + spatial_offset(ch)
          bank = (addr >> byteOffsetWidth) & (num_banks - 1)

        For all current SNAX streamer configurations the spatial
        offsets are multiples of 8 bytes (one bank word), so the bank
        formula simplifies to ``(temporal_base // 8 + ch) % num_banks``.
        """
        counters = self._step_to_counters(step)
        base = sum(self.strides_bank[lvl] * counters[lvl]
                   for lvl in range(self.num_levels))
        return (base // 8 + ch) % self.num_banks

    def _step_to_counters(self, step: int) -> list[int]:
        counters: list[int] = []
        s = step
        for b in self.bounds:
            counters.append(s % b)
            s //= b
        return counters

    def needs_tcdm_access(self, step: int) -> bool:
        """Does this streamer need a TCDM access at ``step``?

        NOTE: only relevant when fixed-cache is modelled.  Without
        fixed-cache every step goes through TCDM.  Kept for the
        legacy ``_simulate_nested`` cost model.
        """
        counters = self._step_to_counters(step)
        if self.kind in (OperandKind.READER, OperandKind.READER_WRITER):
            return _reader_active(counters, self.tiling, self.invariant_dims)
        else:
            return _writer_active(
                counters, self.bounds, self.tiling, self.invariant_dims
            )

    # ----- AGU phase -----

    @property
    def uses_fixed_cache(self) -> bool:
        """Does this streamer use the FixedLevelCache?

        Matches the CriticalLoopFinder RTL logic exactly:

        ``totalBounds(0) = bounds(0)``
        For ``i > 0``:
          if ``strides(i) == 0``: ``totalBounds(i) = totalBounds(i-1)``
          else: ``totalBounds(i) = totalBounds(i-1) * bounds(i)``

        Level ``i`` (where ``i > 0``) is a critical-loop candidate
        when all three hold:
          1. ``totalBounds(i-1) <= fixedCacheDepth``
          2. ``strides(i) == 0``  (stride-0 at that level)
          3. ``bounds(i) > 1``    (non-trivial loop)

        ``anyLoopFound = any candidate``.
        ``enableFixedCache`` is always true for readers and for
        the writer half of a ReaderWriter pair.
        """
        if self._uses_fixed_cache is not None:
            return self._uses_fixed_cache
        # Bound-1 levels are pruned (they are no-ops in hardware),
        # EXCEPT levels whose bound was set to 1 by the repeat
        # override (_repeat_override_level) — those are real.
        eff_bounds = []
        eff_strides = []
        for i in range(self.num_levels):
            if self.bounds[i] > 1 or i == self._repeat_override_level:
                eff_bounds.append(self.bounds[i])
                eff_strides.append(self.strides_bank[i])
        if len(eff_bounds) < 2:
            return False  # need at least 2 effective levels
        # Compute totalBounds exactly as the RTL CriticalLoopFinder
        total_bounds = [0] * len(eff_bounds)
        total_bounds[0] = eff_bounds[0]
        for i in range(1, len(eff_bounds)):
            if eff_strides[i] == 0:
                total_bounds[i] = total_bounds[i - 1]
            else:
                total_bounds[i] = total_bounds[i - 1] * eff_bounds[i]
        # Check each effective level i > 0 for the three conditions
        for i in range(1, len(eff_bounds)):
            if (total_bounds[i - 1] <= self.fixed_cache_depth
                    and eff_strides[i] == 0
                    and eff_bounds[i] > 1):
                return True
        return False

    def agu_tick_possible(self) -> bool:
        """
        Can the AGU advance its counter this cycle?

        Reader with new cache (separate write/read buffers):
          Main counter ticks independently of cache buffers.
          When newUpdateCache: need outputBuffer room (TCDM request).
          Otherwise: tick freely (cache hit, no TCDM request needed).

        Writer with old cache:
          Both outputBuffer and fixedCacheInstructionBuffer must
          have room (unchanged from before).

        Without cache:
          outputBuffer must have room for access steps.
        """
        if self.agu.state != "BUSY":
            return False
        if self.agu.done:
            return False
        ob_has_room = all(
            len(buf) < self.output_buffer_depth
            for buf in self.output_buffers
        )
        if self.uses_fixed_cache and self.is_reader:
            # New reader cache: main counter is independent of cache
            # buffers.  Use _is_update_cache_at (matches RTL
            # newUpdateCache) — NOT needs_tcdm_access which uses
            # the tiling's is_critical flag instead of strides.
            update_cache = self._is_update_cache_at(self.agu.counters)
            if update_cache:
                return ob_has_room
            else:
                return True
        if self.uses_fixed_cache:
            # Writer with old cache
            cib_has_room = (
                len(self.old_cib)
                < self.output_buffer_depth
            )
            return ob_has_room and cib_has_room
        step = self.agu.current_step()
        if self.needs_tcdm_access(step):
            return ob_has_room
        else:
            return True

    def do_agu_tick(self):
        """
        Execute one AGU tick.

        Reader with new cache: push to outputBuffer only for
          updateCache (access) steps.  Cache buffers are populated
          by independent write/read cache counter ticks.
        Writer with old cache: push to cache_instruction_buffer and
          outputBuffer as before.
        Without cache: push to outputBuffer for access steps.
        """
        step = self.agu.current_step()
        is_access = self.needs_tcdm_access(step)
        if self.uses_fixed_cache and self.is_reader:
            # New reader cache: TCDM request on updateCache steps only.
            # Uses _is_update_cache_at (RTL newUpdateCache) not
            # needs_tcdm_access.
            update_cache = self._is_update_cache_at(self.agu.counters)
            if update_cache:
                for buf in self.output_buffers:
                    buf.append(step)
                self.step_grants_remaining[step] = self.spatial_banks
        elif self.uses_fixed_cache:
            # Writer with old cache
            cache_index = self._compute_cache_index_at(self.agu.counters)
            self.cache_instruction_buffer.append((step, is_access, cache_index))
            if is_access:
                for buf in self.output_buffers:
                    buf.append(step)
                self.step_grants_remaining[step] = self.spatial_banks
        else:
            if is_access:
                for buf in self.output_buffers:
                    buf.append(step)
                self.step_grants_remaining[step] = self.spatial_banks
        self.agu.tick()

    # ----- per-channel request / grant helpers -----

    def channel_can_request(self, ch: int, data_fifo_popped: bool = False) -> bool:
        """Can channel ``ch`` issue a new TCDM request this cycle?

        Mirrors the DataRequestor fire conditions:
        * Channel has a valid address (its output buffer queue is
          non-empty).
        * Not already mid-request (``channel_pending_bank[ch]`` is
          None).
        * Reader: DataResponser has room (``responser_slots > 0``)
          **or** ``dataFifoPopped`` is true (combinational bypass
          from DataResponser: ``rspReady = ~lastVal || dataFifoPopped``).
        * Writer: per-channel dataBuffer queue has data to write.
        """
        if self.channel_pending_bank[ch] is not None:
            return False  # already has a pending request
        if not self.output_buffers[ch]:
            return False  # no address in queue
        # Streamer-level preconditions
        if self.kind in (OperandKind.READER, OperandKind.READER_WRITER):
            if self.responser_slots[ch] <= 0 and not data_fifo_popped:
                return False
        else:  # WRITER
            if len(self.writer_channel_bufs[ch]) == 0:
                return False
        return True

    def start_channel_request(self, ch: int):
        """
        Start a TCDM request for channel ``ch``.  Peeks at the head
        of the channel's output buffer queue and computes the target
        bank.
        """
        step = self.output_buffers[ch][0]
        bank = self.compute_channel_bank(step, ch)
        self.channel_pending_bank[ch] = bank
        # Reader: consume this channel's responser slot (reqSubmit fires per-channel)
        if self.kind in (OperandKind.READER, OperandKind.READER_WRITER):
            self.responser_slots[ch] -= 1

    def grant_channel(self, ch: int) -> list[int]:
        """
        Grant channel ``ch``'s bank request.  Dequeues from the
        channel's output buffer queue.

        For writers, also pops from the per-channel data buffer
        (data leaves with the request, matching DataRequestor RTL:
        ``io.in.data.get.ready := io.in.addr.ready``).

        Returns a list of step indices that became fully complete
        (all channels granted) as a result of this grant.
        """
        step = self.output_buffers[ch].popleft()
        self.channel_pending_bank[ch] = None
        # Writer: data is consumed alongside the request (per-channel)
        if self.is_writer:
            self.writer_channel_bufs[ch].popleft()
        completed: list[int] = []
        self.step_grants_remaining[step] -= 1
        if self.step_grants_remaining[step] == 0:
            del self.step_grants_remaining[step]
            completed.append(step)
        return completed

    def has_pending_channels(self) -> bool:
        return any(b is not None for b in self.channel_pending_bank)

    # ----- accelerator interface -----

    def acc_to_agu_step(self, acc_step: int) -> int:
        """Convert a global accelerator step to this streamer's AGU step.

        With repeat_count > 1 the AGU generates fewer steps, each
        presented ``repeat_count`` times by the HandShakeRepeater.
        For repeat_count == 1 this is the identity.
        """
        return acc_step // self.repeat_count

    def is_last_repeat(self, acc_step: int) -> bool:
        """Is ``acc_step`` the last repeat of its AGU step?

        Matches HandShakeRepeater: ``io.in.ready := io.out.fire &&
        dataRepeatCounter.io.lastVal``.
        """
        return acc_step % self.repeat_count == self.repeat_count - 1

    def reader_data_available(self, agu_step: int) -> bool:
        """Does the dataBuffer contain the data for ``agu_step``?"""
        return agu_step in self.data_buffer

    def reader_consume(self, agu_step: int):
        """Pop ``agu_step`` from the dataBuffer (accelerator consumed it)."""
        self.data_buffer.remove(agu_step)
        # dataFifoPopped fires for ALL channels simultaneously
        for ch in range(self.spatial_banks):
            self.responser_slots[ch] += 1

    def writer_has_space(self) -> bool:
        """Can the writer accept a new d_o datum?

        RTL Writer.scala (ReaderWriter, fixed-cache mode):
          instrValid = fixedCacheInstructionBuffer.deq.valid
          goToTCDM  = !instr.useCache || instr.lastAccess
          targetReady = Mux(goToTCDM, dataBuffer.in.head.ready, true.B)
          dataAfterCrosser.ready = instrValid && targetReady

        Without fixed-cache: plain dataBuffer space check.

        The writer's ComplexQueueConcat has ``pipe = false``, so the
        buffer does NOT accept new data in the same cycle that a
        dequeue frees a slot.  We use ``_writer_buf_full_at_start``
        to model this: if ANY per-channel queue was full at the start
        of the cycle, the wide input cannot fire even after grants
        free slots during the cycle.
        """
        if self.uses_fixed_cache:
            if not self.cache_instruction_buffer:
                return False  # no instrValid
            _step, is_access, _cidx = self.cache_instruction_buffer[0]
            if is_access:  # lastAccess → goToTCDM: need dataBuffer room
                return not self._writer_buf_full_at_start
            return True  # cache-only path: no backpressure
        return not self._writer_buf_full_at_start

    def writer_accept(self, acc_step: int):
        """Accelerator pushes result into the writer.

        RTL Writer.scala (ReaderWriter, fixed-cache mode):
          bothValid = dataAfterCrosser.valid && instrValid
          - goToCache (useCache): write to reader's FixedLevelCache
            (combinational, no timing effect in model)
          - goToTCDM (lastAccess): push into dataBuffer for TCDM
            writeback
          Consumes one fixedCacheInstruction per d_o fire.

        Without fixed-cache: push directly to per-channel dataBuffers.
        """
        if self.uses_fixed_cache:
            _step, is_access, cache_index = self.cache_instruction_buffer.popleft()
            if is_access:  # lastAccess → data goes to TCDM
                for buf in self.writer_channel_bufs:
                    buf.append(acc_step)
            else:
                # goToCache: writer drives writerPort into reader's
                # FixedLevelCache memory.  Record the cache index so
                # that the reader's process_reader_cache can detect
                # writerPort bank conflicts on the shared write port.
                self._writer_port_cache_index = cache_index
            return
        for buf in self.writer_channel_bufs:
            buf.append(acc_step)

    # ----- FixedLevelCache processing -----

    def cache_process(self, acc_consumed: bool, writer_port_cache_index: int | None = None):
        """Advance the FixedLevelCache state for one cycle.

        For readers with new separate write/read cache buffers,
        delegates to ``process_reader_cache``.

        For writers with old single cache_instruction_buffer, uses
        the original delayedValid register model.

        Must be called AFTER the accelerator fire decision so that
        ``acc_consumed`` is known.

        Args:
            writer_port_cache_index: for from_rw readers, the cache
                index the paired writer's writerPort writes to this
                cycle (or None if inactive).
        """
        if self.is_reader and self.uses_fixed_cache:
            # New reader cache model with dual-bank FixedLevelCache
            self.process_reader_cache(acc_consumed, writer_port_cache_index)
            return

        # --- Writer / old model ---
        can_accept_new = (self.cache_output_step is None) or acc_consumed
        processed_new = False
        if can_accept_new and self.cache_instruction_buffer:
            step, is_update, _cidx = self.cache_instruction_buffer[0]
            if is_update:
                # Mode 2: needs data from dataBuffer head
                if self.data_buffer:
                    self.cache_instruction_buffer.popleft()
                    self.data_buffer.popleft()
                    # dataFifoPopped: free responser slots for ALL channels
                    for ch in range(self.spatial_banks):
                        self.responser_slots[ch] += 1
                    self.cache_output_step = step
                    processed_new = True
            else:
                # Mode 3: cache hit — read from cache memory, no TCDM
                self.cache_instruction_buffer.popleft()
                self.cache_output_step = step
                processed_new = True
        if not processed_new and acc_consumed:
            self.cache_output_step = None

    @property
    def is_reader(self) -> bool:
        return self.kind in (OperandKind.READER, OperandKind.READER_WRITER)

    @property
    def is_writer(self) -> bool:
        return self.kind == OperandKind.WRITER

    @property
    def is_fully_done(self) -> bool:
        """AGU finished, no pending requests, buffers drained."""
        done = (
            self.agu.done
            and all(len(buf) == 0 for buf in self.output_buffers)
            and not self.has_pending_channels()
            and len(self.step_grants_remaining) == 0
        )
        if self.is_writer:
            done = done and all(len(buf) == 0 for buf in self.writer_channel_bufs)
        else:
            done = done and len(self.data_buffer) == 0
        if self.uses_fixed_cache and self.is_reader:
            # New reader cache state
            done = done and (
                self.write_cache_done
                and self.read_cache_done
                and len(self.write_cache_buffer) == 0
                and len(self.read_cache_buffer) == 0
                and not self.cache_instr_valid
                and self.cache_output_step is None
            )
        elif self.uses_fixed_cache:
            # Writer old cache state
            done = done and (
                len(self.cache_instruction_buffer) == 0
                and self.cache_output_step is None
            )
        return done


# ---------------------------------------------------------------------------
# RW pair tracking
# ---------------------------------------------------------------------------

@dataclass
class ReaderWriterPair:
    """
    In the RTL a ReaderWriter shares one TCDM port set.  The writer
    has priority (MuxDecoupled sel=0 for writer, sel=1 for reader).
    Only one of them can issue requests on a given cycle.

    ``reader_idx`` and ``writer_idx`` index into the ``streamers``
    list.
    """
    reader_idx: int
    writer_idx: int
    # Track which side was selected last cycle (for response routing)
    last_sel_writer: bool = False


# ---------------------------------------------------------------------------
# Sparse Interconnect Arbiter — cycle-accurate model of per-bank
# round-robin arbitration matching the RTL SparseInterconnect,
# ArbitrationTree, and RoundRobinArbiter modules.
# ---------------------------------------------------------------------------

# Full SNAX gemmx sparse interconnect configuration including system ports.
# Matches the RTL generated config from snaxgen.py:
#   [[8,1],[8,8],[8,8],[32,8],[16,1],[1,1],[1,1],[1,1],[8,8]]
# (accelerator ports 0-3 + xDMA + core0 + core1 + AXI + iDMA)
GEMMX_SPARSE_PORT_CONFIG: list["SparsePortDef"] = None  # type: ignore[assignment]  # set after class def
GEMMX_DORMANT_PORTS: frozenset[int] = frozenset({2})  # Port 2 (D standalone writer) is dormant

@dataclass(frozen=True)
class SparsePortDef:
    """Mirrors the RTL ``SparsePortDefinition(width, access_granularity)``."""
    width: int
    access_granularity: int

    @property
    def inputs_per_bank(self) -> int:
        return self.width // self.access_granularity


# Now that SparsePortDef is defined, set the gemmx constant
GEMMX_SPARSE_PORT_CONFIG = [
    SparsePortDef(8, 1),    # Port 0 — A reader
    SparsePortDef(8, 8),    # Port 1 — B reader
    SparsePortDef(8, 8),    # Port 2 — D standalone writer (dormant)
    SparsePortDef(32, 8),   # Port 3 — C/D ReaderWriter
    SparsePortDef(16, 1),   # Port 4 — xDMA
    SparsePortDef(1, 1),    # Port 5 — core 0
    SparsePortDef(1, 1),    # Port 6 — core 1
    SparsePortDef(1, 1),    # Port 7 — AXI
    SparsePortDef(8, 8),    # Port 8 — iDMA
]


@dataclass
class SparseInterconnectConfig:
    """
    Mirrors the RTL ``SparseConfig``.

    Given the port definitions and total bank count, pre-computes the
    mapping from (streamer_index, channel) pairs to global TCDM input
    indices, and the per-bank mapping from arbiter-local input index
    to global input index.

    The global input ordering matches the RTL: ports are concatenated
    in order, and within each port the channels are laid out by their
    ``width``.  Streamers are assigned to ports in the order they
    appear (accounting for RW splits sharing one port).
    """
    ports: list[SparsePortDef]
    num_banks: int

    # --- Derived (computed in __post_init__) ---
    # Total inputs across all ports
    total_inputs: int = field(init=False)
    # inputs_per_bank: how many wires feed into each bank's arbiter
    inputs_per_bank: int = field(init=False)
    # per_bank_global_idx[bank][local_idx] → global input index
    per_bank_global_idx: list[list[int]] = field(init=False, repr=False)

    def __post_init__(self):
        self.total_inputs = sum(p.width for p in self.ports)
        self.inputs_per_bank = sum(p.inputs_per_bank for p in self.ports)
        # Replicate the RTL get_global_idx_list(bank) logic
        self.per_bank_global_idx = []
        for bank in range(self.num_banks):
            idx_list: list[int] = []
            acc = 0
            for p in self.ports:
                for i in range(p.inputs_per_bank):
                    idx_list.append(
                        acc + i * p.access_granularity + bank % p.access_granularity
                    )
                acc += p.width
            self.per_bank_global_idx.append(idx_list)

    def global_to_local(self, bank: int, global_idx: int) -> int | None:
        """Return the local arbiter input index for *global_idx* on *bank*,
        or None if that global input is not wired to this bank."""
        try:
            return self.per_bank_global_idx[bank].index(global_idx)
        except ValueError:
            return None


@dataclass
class SparseInterconnectArbiter:
    """
    Cycle-accurate model of the SNAX SparseInterconnect.

    Each of the *num_banks* memory banks has its own ``ArbitrationTree``
    containing a ``RoundRobinArbiter(inputs_per_bank)``.  The arbiter
    selects **at most one** winner per cycle per bank using round-robin
    arbitration.

    RoundRobinArbiter RTL behaviour (with memReq.ready always true):
      * ``previous`` register = RegNext(selection.bits), init = NumInp-1
      * ``lock`` is always false (ready is always true)
      * Selection:
        1. Find valid requests with index > previous (``nextRequests``)
        2. If any → PriorityEncoder(nextRequests) wins
        3. Else → PriorityEncoder(allValidRequests) wins (wrap-around)
      * ``previous`` updates to the winning local index every cycle
        where any request was valid — even if the same winner is
        selected.  When no request is valid, ``previous`` holds its
        value (RegNext of the combinational output, but output is
        ``anyValid``-gated — actually the Chisel code unconditionally
        does ``RegNext(io.selection.bits)``, so ``previous`` always
        latches the last combinational ``selectedRequest``).

    Parameters
    ----------
    config : SparseInterconnectConfig
        Pre-computed port/bank mapping.
    streamer_channel_to_global : dict[(int, int), int]
        Maps (streamer_idx, channel_idx) to a global TCDM input index.
        Built externally when the streamer list is constructed.
    """
    config: SparseInterconnectConfig
    streamer_channel_to_global: dict[tuple[int, int], int]

    # Per-bank ``previous`` register, init to inputs_per_bank - 1.
    _bank_previous: list[int] = field(init=False, repr=False)

    def __post_init__(self):
        n = self.config.inputs_per_bank
        self._bank_previous = [n - 1] * self.config.num_banks

    def arbitrate(
        self,
        requests: dict[int, list[tuple[int, int]]],
    ) -> list[tuple[int, int, int]]:
        """
        Run one cycle of per-bank round-robin arbitration.

        Parameters
        ----------
        requests : dict[bank, list[(streamer_idx, channel_idx)]]
            All pending TCDM requests this cycle, grouped by target
            bank.  Each entry is a (streamer_idx, channel) pair.

        Returns
        -------
        grants : list[(streamer_idx, channel_idx, bank)]
            The granted requests this cycle (at most one per bank).
        """
        grants: list[tuple[int, int, int]] = []
        ipb = self.config.inputs_per_bank

        # Track which banks had a valid winner this cycle.
        banks_with_winner: set[int] = set()

        for bank, requestors in requests.items():
            if not requestors:
                continue

            # Map each requestor to its local arbiter input index
            local_requests: list[tuple[int, int, int]] = []  # (local_idx, si, ch)
            for si, ch in requestors:
                global_idx = self.streamer_channel_to_global.get((si, ch))
                if global_idx is None:
                    continue
                local_idx = self.config.global_to_local(bank, global_idx)
                if local_idx is None:
                    continue
                local_requests.append((local_idx, si, ch))

            if not local_requests:
                continue

            # --- Round-robin selection (mirrors RoundRobinArbiter RTL) ---
            previous = self._bank_previous[bank]

            # nextRequests: valid requests with local_idx > previous
            next_reqs = [(li, si, ch) for li, si, ch in local_requests
                         if li > previous]

            if next_reqs:
                # PriorityEncoder: pick smallest local_idx among next_reqs
                winner = min(next_reqs, key=lambda x: x[0])
            else:
                # Wrap around: PriorityEncoder of all valid requests
                winner = min(local_requests, key=lambda x: x[0])

            local_idx, si, ch = winner
            grants.append((si, ch, bank))
            banks_with_winner.add(bank)

            # Update previous register (RegNext of selection.bits)
            self._bank_previous[bank] = local_idx

        # Banks with no valid requests: RTL's PriorityEncoder(all_false)
        # returns NumInp-1 (Chisel PriorityMux recurses to the last
        # element when no input is true), so previous latches ipb-1.
        for bank in range(self.config.num_banks):
            if bank not in banks_with_winner:
                self._bank_previous[bank] = ipb - 1

        return grants


def build_sparse_interconnect(
    streamers: list,
    rw_pairs: list,
    num_banks: int,
    sparse_port_config: list[SparsePortDef] | None = None,
    dormant_ports: frozenset[int] | None = None,
) -> SparseInterconnectArbiter:
    """
    Build a ``SparseInterconnectArbiter`` from the simulation's
    streamer list.

    The default sparse config for the SNAX gemmx accelerator (matching
    ``fixed_cache_test_32_banks.hjson``) is::

        [[8, 1], [8, 8], [8, 8], [32, 8]]

    Port assignment (order must match RTL):
      Port 0 — A reader (8 channels, access_granularity=1)
      Port 1 — B reader (8 channels, access_granularity=8)
      Port 2 — D writer (8 channels, access_granularity=8)
      Port 3 — C/D RW  (32 channels, access_granularity=8)

    For RW pairs the reader and writer halves share one set of TCDM
    channels (the MuxDecoupled selects between them each cycle).
    Both halves map to the same global input indices.
    """
    if dormant_ports is None:
        dormant_ports = frozenset()

    if sparse_port_config is None:
        # Auto-generate port config from the actual streamers.
        # Non-RW streamers get one port each; each RW pair gets one port.
        # access_granularity defaults to 1 (any bank addressable) for
        # generic configs; pass an explicit sparse_port_config for
        # hardware-specific mappings (e.g. gemmx).
        rw_reader_ids = {p.reader_idx for p in rw_pairs}
        rw_writer_ids = {p.writer_idx for p in rw_pairs}
        sparse_port_config = []
        for si, s in enumerate(streamers):
            if si in rw_reader_ids or si in rw_writer_ids:
                continue
            sparse_port_config.append(SparsePortDef(s.spatial_banks, 1))
        for pair in rw_pairs:
            sparse_port_config.append(
                SparsePortDef(streamers[pair.reader_idx].spatial_banks, 1)
            )

    config = SparseInterconnectConfig(ports=sparse_port_config, num_banks=num_banks)

    # Build (streamer_idx, channel) → global input index mapping.
    # Ports are assigned in order: first non-RW streamers in order,
    # then RW pairs.  Within each port, channel ``ch`` maps to
    # global index = port_base + ch.
    channel_to_global: dict[tuple[int, int], int] = {}

    # Identify RW reader/writer indices
    rw_reader_set = {p.reader_idx for p in rw_pairs}
    rw_writer_set = {p.writer_idx for p in rw_pairs}

    global_base = 0
    port_idx = 0

    def _skip_dormant():
        """Advance past dormant ports, accumulating their global width."""
        nonlocal global_base, port_idx
        while port_idx in dormant_ports and port_idx < len(sparse_port_config):
            global_base += sparse_port_config[port_idx].width
            port_idx += 1

    # Non-RW streamers first (in order of streamer index)
    for si, s in enumerate(streamers):
        if si in rw_reader_set or si in rw_writer_set:
            continue
        _skip_dormant()
        if port_idx >= len(sparse_port_config):
            break
        p = sparse_port_config[port_idx]
        for ch in range(s.spatial_banks):
            channel_to_global[(si, ch)] = global_base + ch
        global_base += p.width
        port_idx += 1

    # RW pairs: reader and writer share the same global indices
    for pair in rw_pairs:
        _skip_dormant()
        if port_idx >= len(sparse_port_config):
            break
        p = sparse_port_config[port_idx]
        reader_s = streamers[pair.reader_idx]
        for ch in range(reader_s.spatial_banks):
            channel_to_global[(pair.reader_idx, ch)] = global_base + ch
            channel_to_global[(pair.writer_idx, ch)] = global_base + ch
        global_base += p.width
        port_idx += 1

    return SparseInterconnectArbiter(
        config=config,
        streamer_channel_to_global=channel_to_global,
    )


@dataclass
class BlockGemmState:
    """
    Models the BlockGemm accelerator state machine with independent
    per-streamer ready/valid signals, matching the RTL in
    BlockGemm.scala and GemmTileArray.scala.

    Ports (mapped to streamer indices):
      - a_i (Flipped Decoupled): reader for operand A
      - b_i (Flipped Decoupled): reader for operand B
      - c_i (Flipped Decoupled): reader for operand C (accumulator input)
      - d_o (Decoupled): writer for operand D (result output)

    Key RTL behaviours modelled:
      1. DecoupledCat4to1 synchronises a_i and b_i — both must be
         valid simultaneously for the combined value to enter the
         register cut.
      2. Register cut (``-\\>``) on the combined a+b bus: 1-cycle
         pipeline delay before data reaches the compute array.
      3. compute_fire_counter: counts 0..K-1.  When 0 the tile
         additionally needs c_i.  When K-1 the output accumulation
         is complete (result available next cycle).
      4. Tile back-pressure:
           d_valid_o = data_i_fire_reg || keep_output
           a_b_c_ready_o = !keep_output && !(d_valid_o && !d_ready_i)
           keep_output (reg) = d_valid_o && !d_ready_i
      5. d_o.valid: asserted when d_output_ifvalid_counter == K-1
         and d_valid_o is true.
      6. d_ready_i = Mux(d_o.valid, d_o.ready, 1.B) — always
         accepts internally when d_o is not externally valid.
    """

    # --- Configuration ---
    K: int  # reduction dimension (number of a+b computes per output tile)
    M_N: int  # total output tiles (M * N)
    a_streamer_idx: int  # index into streamers[] for a_i
    b_streamer_idx: int  # index into streamers[] for b_i
    c_streamer_idx: int  # index into streamers[] for c_i (reader half of RW)
    d_streamer_idx: int  # index into streamers[] for d_o (writer half of RW)

    # --- Runtime state (registers) ---
    busy: bool = False
    compute_fire_counter: int = 0  # 0..K-1
    d_output_ifvalid_counter: int = 0  # counts K output fires
    d_output_counter: int = 0  # counts M*N total output writes

    # DataCut(delay=2) pipeline: models the -\\> operator used in
    # BlockGemm.scala for combined_decoupled_a_b_in -\\> combined_decoupled_a_b_out.
    # This is a 2-stage shift register with enable.
    #   shiftPermission = (outValid && out.ready) || !outValid
    #   io.in.ready = shiftPermission
    #   shift = shiftPermission && shiftSuggestion
    # When shift fires, all stages advance simultaneously.
    # stage[1] is the output (connected to combined_decoupled_a_b_out).
    datacut_stage0_valid: bool = False
    datacut_stage1_valid: bool = False  # = combined_decoupled_a_b_out.valid

    # Tile registered state
    data_i_fire_reg: bool = False  # compute fired last cycle → d_valid_o
    keep_output: bool = False  # back-pressure: output not consumed

    # Track total accelerator steps consumed (for compatibility with
    # the rest of the simulation which uses ``acc_step``).
    acc_step: int = 0  # compute-side step (which step the tile is processing)
    input_step: int = 0  # regcut-input-side step (which step a+b are being fetched for)

    # --- Per-streamer consumed flags (set by tick(), read by simulation) ---
    # True when that reader's data was actually popped from its
    # dataBuffer (or cache output consumed) this cycle.
    # The agu_step fields store which agu_step was consumed (only
    # valid when the corresponding bool is True).
    a_consumed: bool = False
    a_consumed_agu_step: int = -1
    b_consumed: bool = False
    b_consumed_agu_step: int = -1
    c_consumed: bool = False
    c_consumed_agu_step: int = -1
    # True when the compute array fired (regcut output consumed)
    compute_fired: bool = False
    # True when d_o fired (output pushed to writer)
    d_produced: bool = False

    # --- Combinational signals (recomputed each cycle) ---

    def _d_valid_o(self) -> bool:
        """Tile's d_valid_o = data_i_fire_reg || keep_output"""
        return self.data_i_fire_reg or self.keep_output

    def _a_b_c_ready_o(self, d_ready_i: bool) -> bool:
        """Tile: a_b_c_ready_o = !keep_output && !(d_valid_o && !d_ready_i)"""
        d_valid = self._d_valid_o()
        return (not self.keep_output) and not (d_valid and not d_ready_i)

    def _d_o_valid(self) -> bool:
        """BlockGemm d_o.valid output to the writer streamer.

        When K==1: d_valid_o && busy
        Otherwise: (d_output_ifvalid_counter == K-1) && d_valid_o && busy
        """
        if not self.busy:
            return False
        d_valid = self._d_valid_o()
        if self.K == 1:
            return d_valid
        return (self.d_output_ifvalid_counter == self.K - 1) and d_valid

    def tick(self, streamers: list) -> bool:
        """
        Advance the BlockGemm state machine by one cycle.

        This replaces the monolithic ``acc_can_fire`` logic.  Instead
        of requiring all streamers to be ready simultaneously, it
        evaluates each port independently based on the RTL handshake
        protocol.

        Returns True if the accelerator consumed at least one input
        (any fire happened), for use in Phase 5 cache logic.

        Call order within the simulation loop:
          Phase 4a: tick() — evaluates all combinational signals,
            updates registered state for next cycle.
        """
        # Reset per-cycle flags
        self.a_consumed = False
        self.a_consumed_agu_step = -1
        self.b_consumed = False
        self.b_consumed_agu_step = -1
        self.c_consumed = False
        self.c_consumed_agu_step = -1
        self.compute_fired = False
        self.d_produced = False

        if not self.busy:
            return False

        s_a: StreamerState = streamers[self.a_streamer_idx]
        s_b: StreamerState = streamers[self.b_streamer_idx]
        s_c: StreamerState = streamers[self.c_streamer_idx]
        s_d: StreamerState = streamers[self.d_streamer_idx]

        # --- Step 1: Determine d_ready_i (writer back-pressure) ---
        # d_ready_i = Mux(d_o.valid, d_o.ready, 1.B)
        # d_o.ready = writer streamer has space in data_buffer
        d_o_valid = self._d_o_valid()
        d_o_ready = s_d.writer_has_space()
        d_ready_i = d_o_ready if d_o_valid else True

        # --- Step 2: Tile ready signal ---
        a_b_c_ready = self._a_b_c_ready_o(d_ready_i)

        # --- Step 3: combined_decoupled_a_b_out (after DataCut) ---
        # stage[1] of the DataCut is the output.
        combined_out_valid = self.datacut_stage1_valid

        # --- Step 4: a_b_data_valid ---
        # At counter==0: needs combined_out + c_i valid
        # Otherwise: just combined_out valid
        agu_step_a = s_a.acc_to_agu_step(self.acc_step)
        agu_step_b = s_b.acc_to_agu_step(self.acc_step)

        c_valid = False
        if self.compute_fire_counter == 0:
            agu_step_c = s_c.acc_to_agu_step(self.acc_step) // self.K
            c_valid = self._reader_data_valid(s_c, agu_step_c)
            a_b_data_valid = combined_out_valid and c_valid
        else:
            a_b_data_valid = combined_out_valid

        # --- Step 5: Compute fire ---
        # gemm_a_b_input_fire = a_b_data_ready && a_b_data_valid
        # a_b_data_ready = a_b_c_ready && busy
        a_b_data_ready = a_b_c_ready and self.busy
        gemm_input_fire = a_b_data_ready and a_b_data_valid

        # --- Step 6: add_c logic ---
        add_c = (self.compute_fire_counter == 0 and self.busy
                 and c_valid and a_b_data_valid)
        add_c_fire = add_c and a_b_c_ready

        # --- Step 7: d_o fire (output to writer streamer) ---
        d_o_fire = d_o_valid and d_o_ready

        # --- Step 8: gemm_output_fire (internal array output) ---
        d_valid_o = self._d_valid_o()
        gemm_output_fire = d_valid_o and d_ready_i

        # ============================================================
        # Consume from / produce to streamers
        # ============================================================
        any_consumed = False

        # Output: if d_o fires, push to writer data_buffer
        computation_finish = False
        if d_o_fire:
            s_d.writer_accept(self.acc_step - 1)  # the step that produced this result
            # Check computation finish BEFORE incrementing (matches RTL)
            computation_finish = (self.d_output_counter == self.M_N - 1)
            self.d_output_counter += 1
            self.d_produced = True

        # Input consumption:
        if gemm_input_fire:
            any_consumed = True
            self.compute_fired = True

            # Consume c if add_c_fire
            if add_c_fire:
                if s_c.is_last_repeat(self.acc_step):
                    self.c_consumed = True
                    self.c_consumed_agu_step = s_c.acc_to_agu_step(self.acc_step) // self.K

        # ============================================================
        # DataCut(delay=2) INPUT: can a+b enter the pipeline?
        # ============================================================
        # The DataCut is a 2-stage shift register from the -\\>
        # operator in BlockGemm.scala.
        #   shiftPermission = (outValid && out.ready) || !outValid
        #   io.in.ready = shiftPermission
        # When the output is being consumed (gemm_input_fire acts as
        # out.ready for the compute side), or when the output is not
        # valid, the pipeline can accept new data.
        #
        # RTL: combined_decoupled_a_b_out.ready :=
        #        cstate === sBUSY && gemm_a_b_input_fire
        # gemm_a_b_input_fire = a_b_data_ready && a_b_data_valid
        # So out.ready requires BOTH the tile ready AND data valid
        # (which at counter==0 includes c_i.valid).
        datacut_out_ready = gemm_input_fire
        datacut_out_valid = self.datacut_stage1_valid
        shift_permission = (datacut_out_valid and datacut_out_ready) or (not datacut_out_valid)
        # shiftSuggestion = dataInsideShiftRegister || in.valid
        # We track dataInsideShiftRegister via stage0_valid.
        datacut_input_ready = shift_permission

        # The input side tracks its own step counter (input_step),
        # which can be up to 1 ahead of acc_step (the compute side)
        # because the regcut decouples them.
        if self.input_step < (self.M_N * self.K):
            next_agu_a = s_a.acc_to_agu_step(self.input_step)
            next_agu_b = s_b.acc_to_agu_step(self.input_step)
            a_valid = self._reader_data_valid(s_a, next_agu_a)
            b_valid = self._reader_data_valid(s_b, next_agu_b)
        else:
            a_valid = False
            b_valid = False

        # DecoupledCat4to1 synchronises a_i, b_i, subtraction_a,
        # subtraction_b.  All four inputs must be valid before the
        # combined output is valid (and thus before any input ready
        # can be asserted).
        #   RTL: decoupled_subtraction_a.valid := cstate === sBUSY
        #        decoupled_subtraction_b.valid := cstate === sBUSY
        #   RTL: io.out.valid := in1.valid && in2.valid && in3.valid && in4.valid
        #        io.inN.ready := io.out.ready && io.out.valid
        sa_valid = self.busy
        sb_valid = self.busy
        cat4to1_out_valid = a_valid and b_valid and sa_valid and sb_valid

        regcut_input_fire = datacut_input_ready and cat4to1_out_valid

        # Consume a and b from streamers when regcut input fires
        if regcut_input_fire:
            if s_a.is_last_repeat(self.input_step):
                self.a_consumed = True
                self.a_consumed_agu_step = s_a.acc_to_agu_step(self.input_step)
            if s_b.is_last_repeat(self.input_step):
                self.b_consumed = True
                self.b_consumed_agu_step = s_b.acc_to_agu_step(self.input_step)
            self.input_step += 1

        # ============================================================
        # Update registered state for next cycle
        # ============================================================

        # compute_fire_counter update
        if gemm_input_fire:
            if self.K == 1:
                pass  # stays at 0
            elif self.compute_fire_counter == self.K - 1:
                self.compute_fire_counter = 0
            else:
                self.compute_fire_counter += 1
            self.acc_step += 1

        # d_output_ifvalid_counter update
        if gemm_output_fire:
            if self.K == 1:
                pass  # stays at 0
            elif self.d_output_ifvalid_counter == self.K - 1:
                self.d_output_ifvalid_counter = 0
            else:
                self.d_output_ifvalid_counter += 1

        # Tile registers
        new_data_i_fire_reg = gemm_input_fire
        new_keep_output = d_valid_o and (not d_ready_i)

        # DataCut shift register update
        # shift = shiftPermission && shiftSuggestion
        # shiftSuggestion = dataInsideShiftRegister || io.in.valid
        # dataInsideShiftRegister tracks whether ANY stage has data
        # (RTL uses a counter: insideCounter != delay).  For delay=2
        # this is equivalent to stage0_valid || stage1_valid.
        # io.in.valid = DecoupledCat4to1 output valid
        #             = a.valid && b.valid && sa.valid && sb.valid
        data_inside = self.datacut_stage0_valid or self.datacut_stage1_valid
        shift_suggestion = data_inside or cat4to1_out_valid
        shift = shift_permission and shift_suggestion

        if shift:
            new_datacut_stage1_valid = self.datacut_stage0_valid
            new_datacut_stage0_valid = regcut_input_fire
        else:
            new_datacut_stage1_valid = self.datacut_stage1_valid
            new_datacut_stage0_valid = self.datacut_stage0_valid

        # Apply register updates
        self.data_i_fire_reg = new_data_i_fire_reg
        self.keep_output = new_keep_output
        self.datacut_stage0_valid = new_datacut_stage0_valid
        self.datacut_stage1_valid = new_datacut_stage1_valid

        # Check computation finish
        if d_o_fire and computation_finish and self.busy:
            self.busy = False

        return any_consumed

    def _reader_data_valid(self, s: 'StreamerState', agu_step: int) -> bool:
        """Check if a reader streamer has data available."""
        if s.uses_fixed_cache:
            return s.cache_output_step == agu_step
        else:
            if s.needs_tcdm_access(agu_step):
                return s.reader_data_available(agu_step)
            return True  # invariant step — always "valid"

    def _predict_acc_consumes_reader(self, si: int, streamers: list) -> bool:
        """Predict whether tick() will set the consumed flag for reader si.

        Mirrors the exact conditions in tick() that set
        a_consumed / b_consumed / c_consumed.
        """
        s = streamers[si]

        if si == self.a_streamer_idx or si == self.b_streamer_idx:
            # a/b: consumed when datacut_input_fire AND is_last_repeat.
            # datacut_input_ready = shift_permission
            #   = (outValid && out.ready) || !outValid
            # RTL: out.ready = sBUSY && gemm_a_b_input_fire
            #   = a_b_data_ready && a_b_data_valid
            # So we need to compute gemm_input_fire to get shift_permission.
            d_o_valid = self._d_o_valid()
            d_o_ready = streamers[self.d_streamer_idx].writer_has_space()
            d_ready_i = d_o_ready if d_o_valid else True
            a_b_c_ready = self._a_b_c_ready_o(d_ready_i)
            a_b_data_ready = a_b_c_ready and self.busy

            # Compute a_b_data_valid (same logic as tick Step 4)
            combined_out_valid = self.datacut_stage1_valid
            s_c = streamers[self.c_streamer_idx]
            if self.compute_fire_counter == 0:
                agu_step_c = s_c.acc_to_agu_step(self.acc_step) //self.K
                c_valid = self._reader_data_valid(s_c, agu_step_c)
                a_b_data_valid = combined_out_valid and c_valid
            else:
                a_b_data_valid = combined_out_valid

            gemm_input_fire = a_b_data_ready and a_b_data_valid
            datacut_out_valid = self.datacut_stage1_valid
            shift_permission = (
                (datacut_out_valid and gemm_input_fire) or
                (not datacut_out_valid)
            )
            if not shift_permission:
                return False
            step = self.input_step
            if step >= self.M_N * self.K:
                return False
            if not s.is_last_repeat(step):
                return False
            # Both a AND b (and subtraction via Cat4to1) must be valid
            # for regcut input to fire.
            s_a = streamers[self.a_streamer_idx]
            s_b = streamers[self.b_streamer_idx]
            a_agu = s_a.acc_to_agu_step(step)
            b_agu = s_b.acc_to_agu_step(step)
            sa_valid = self.busy
            sb_valid = self.busy
            return (self._reader_data_valid(s_a, a_agu) and
                    self._reader_data_valid(s_b, b_agu) and
                    sa_valid and sb_valid)

        elif si == self.c_streamer_idx:
            # c: consumed when add_c_fire AND is_last_repeat
            if self.compute_fire_counter != 0:
                return False
            if not self.datacut_stage1_valid:
                return False  # combined_out_valid required
            step = self.acc_step
            if not s.is_last_repeat(step):
                return False
            agu_step = s.acc_to_agu_step(step) // self.K
            c_valid = self._reader_data_valid(s, agu_step)
            if not c_valid:
                return False
            # a_b_c_ready_o: check tile back-pressure
            d_o_valid = self._d_o_valid()
            d_o_ready = streamers[self.d_streamer_idx].writer_has_space()
            d_ready_i = d_o_ready if d_o_valid else True
            return self._a_b_c_ready_o(d_ready_i)

        return False

    def predict_data_fifo_popped(self, si: int, streamers: list) -> bool:
        """Predict whether reader si's dataBuffer will be popped this cycle.

        Unified prediction for ALL reader types (cache and non-cache).
        Used for the dataFifoPopped combinational bypass in Phase 2.

        For new-model cache readers (separate write/read buffers):
          dataBuffer popped when FixedLevelCache accepts a write
          instruction.  This requires write_cache_buffer + data_buffer
          both non-empty, and no bank conflict with a simultaneous
          read instruction.

        For old-model cache readers (single instruction buffer):
          dataBuffer popped when cache_process handles an updateCache
          instruction with data.

        For non-cache readers: dataBuffer popped directly by the
          accelerator when the step needs TCDM access.
        """

        s = streamers[si]

        if s.uses_fixed_cache and s.is_reader:
            # New reader cache: dataFifoPopped when FixedLevelCache
            # accepts a write (write_cache_buffer + data_buffer valid
            # AND no bank conflict).
            if not s.write_cache_buffer or not s.data_buffer:
                return False
            write_index = s.write_cache_buffer[0]
            write_bank = write_index & 1

            if s.from_rw:
                # Dual-port SRAM: read port independent of write port.
                # Write blocked only by writerPort on same bank.
                # Predict whether d_o fires with goToCache this cycle.
                writer_port_bank = self._predict_writer_port_bank(s, streamers)
                if writer_port_bank is not None and write_bank == writer_port_bank:
                    return False  # writerPort conflict, write stalls
                # Also check read/write same-address conflict on
                # dual-port SRAM (same bank AND same address).
                acc_consumed = self._predict_acc_consumes_reader(si, streamers)
                data_avail = s._cache_data_available()
                delivering = data_avail and acc_consumed
                can_accept_new = (not s.cache_instr_valid) or delivering
                read_issued = can_accept_new and bool(s.read_cache_buffer)
                if read_issued:
                    read_index, _step, _la = s.read_cache_buffer[0]
                    read_bank = read_index & 1
                    write_addr = write_index >> 1
                    read_addr = read_index >> 1
                    if read_bank == write_bank and read_addr == write_addr:
                        return False  # same-address conflict, write stalls
            else:
                # Single-port SRAM: read has priority over write.
                # Predict whether a read will be accepted this cycle
                # (determines bank conflict for write).
                acc_consumed = self._predict_acc_consumes_reader(si, streamers)
                data_avail = s._cache_data_available()
                delivering = data_avail and acc_consumed
                can_accept_new = (not s.cache_instr_valid) or delivering
                read_issued = can_accept_new and bool(s.read_cache_buffer)
                if read_issued:
                    read_index, _step, _la = s.read_cache_buffer[0]
                    read_bank = read_index & 1
                    if write_bank == read_bank:
                        return False  # bank conflict, write stalls
            return True
        elif s.uses_fixed_cache:
            # Old-model cache (writer streamers that use cache)
            acc_consumed = self._predict_acc_consumes_reader(si, streamers)
            can_accept = (s.cache_output_step is None) or acc_consumed
            if can_accept and s.cache_instruction_buffer:
                step, is_update, _cidx = s.cache_instruction_buffer[0]
                return is_update and bool(s.data_buffer)
            return False
        else:
            # Non-cache: direct buffer pop when acc consumes AND
            # this step actually has a buffer entry.
            if not self._predict_acc_consumes_reader(si, streamers):
                return False
            if si == self.a_streamer_idx or si == self.b_streamer_idx:
                step = self.input_step
            elif si == self.c_streamer_idx:
                step = self.acc_step
            else:
                return False
            agu_step = s.acc_to_agu_step(step)
            return s.needs_tcdm_access(agu_step)

    def _predict_writer_port_bank(self, reader_s, streamers: list) -> int | None:
        """Predict the bank the writerPort writes to this cycle.

        Returns the bank index (0 or 1) if the paired writer fires
        with goToCache, or None if the writerPort is inactive.

        The writerPort fires when d_o fires AND the writer's head
        cache instruction is goToCache (is_access=False).
        """
        if reader_s._paired_writer_idx is None:
            return None
        s_d = streamers[reader_s._paired_writer_idx]
        if not s_d.uses_fixed_cache or not s_d.cache_instruction_buffer:
            return None
        _step, is_access, cache_index = s_d.cache_instruction_buffer[0]
        if is_access:
            return None  # goToTCDM, writerPort not active
        # Predict d_o fire: d_o_valid AND d_o_ready
        d_o_valid = self._d_o_valid()
        d_o_ready = s_d.writer_has_space()
        if not (d_o_valid and d_o_ready):
            return None
        return cache_index & 1

    @property
    def is_done(self) -> bool:
        """Has the accelerator finished all computation?"""
        return not self.busy and self.d_output_counter >= self.M_N


def _simulate_hardware(
    tiling: list[tuple[int, int, bool]],
    operand_descriptors: Sequence[OperandDescriptor],
    invariance_map: list[set[int]],
    strides_bank: list[list[int]],
    num_banks: int,
    *,
    sparse_port_config: "list[SparsePortDef] | None" = None,
    dormant_ports: frozenset[int] | None = None,
) -> int:
    """
    Cycle-accurate simulation of the SNAX streamer hardware pipeline.

    Each streamer is modelled as a ``StreamerState`` object that
    mirrors the RTL modules (AGU, outputBuffer, DataRequestors,
    DataResponsers, dataBuffer).  ReaderWriter operands are split
    into separate reader and writer ``StreamerState`` instances that
    share TCDM ports through priority arbitration (writer wins).

    Per-cycle phases
    ----------------
    0. **AGU phase** — each AGU that is busy and whose outputBuffer
       has room generates one address-set (counter tick).
    1. **Request issue** — streamers with addresses in their
       outputBuffer and meeting preconditions (reader: responser has
       room; writer: dataBuffer has data) start per-channel TCDM
       requests.  RW pairs arbitrate (writer priority).
    2. **Bank arbitration** — per-bank round-robin among all channel
       requests from all streamers.  Each bank services at most one
       request per cycle.
    3. **Burst completion** — streamers whose *all* channels have
       been granted complete their burst (reader: push to dataBuffer;
       writer: pop from dataBuffer).
    4. **Accelerator fire** — the accelerator fires if every reader
       has its data ready and every writer has space.  On fire:
       readers consume, writers accept.
    5. **Commit** — next-state becomes current state.

    Returns total cycle count.
    """
    num_levels = len(tiling)
    if num_levels == 0:
        return 0

    bounds = [t[1] for t in tiling]
    total_steps = 1
    for b in bounds:
        total_steps *= b

    # --- Build StreamerState list, splitting RW into reader+writer ---

    streamers: list[StreamerState] = []
    rw_pairs: list[ReaderWriterPair] = []
    # Track which streamers share TCDM ports (RW pairs)
    # All other streamers have their own dedicated TCDM ports

    cycle_for_step_i = [0] * total_steps  # for debugging: cycle count when accelerator step i fires

    for op_idx, desc in enumerate(operand_descriptors):
        if desc.kind == OperandKind.READER_WRITER:
            # Split into a reader streamer and a writer streamer
            reader_s = StreamerState(
                op_idx=op_idx,
                kind=OperandKind.READER,
                spatial_banks=desc.spatial_banks,
                invariant_dims=desc.invariant_dims,
                num_levels=num_levels,
                bounds=list(bounds),
                strides_bank=list(strides_bank[op_idx]),
                num_banks=num_banks,
                tiling=tiling,
                from_rw=True,
                fixed_cache_depth=desc.fixed_cache_depth,
            )
            reader_s.agu.reset_and_start()
            reader_idx = len(streamers)
            streamers.append(reader_s)

            writer_s = StreamerState(
                op_idx=op_idx,
                kind=OperandKind.WRITER,
                spatial_banks=desc.spatial_banks,
                invariant_dims=desc.invariant_dims,
                num_levels=num_levels,
                bounds=list(bounds),
                strides_bank=list(strides_bank[op_idx]),
                num_banks=num_banks,
                tiling=tiling,
                from_rw=True,
                fixed_cache_depth=desc.fixed_cache_depth,
            )
            writer_s.agu.reset_and_start()
            writer_idx = len(streamers)
            streamers.append(writer_s)

            rw_pairs.append(ReaderWriterPair(
                reader_idx=reader_idx,
                writer_idx=writer_idx,
            ))

            # Link reader to its paired writer for writerPort prediction
            reader_s._paired_writer_idx = writer_idx
        else:
            s = StreamerState(
                op_idx=op_idx,
                kind=desc.kind,
                spatial_banks=desc.spatial_banks,
                invariant_dims=desc.invariant_dims,
                num_levels=num_levels,
                bounds=list(bounds),
                strides_bank=list(strides_bank[op_idx]),
                num_banks=num_banks,
                tiling=tiling,
                fixed_cache_depth=desc.fixed_cache_depth,
            )
            s.agu.reset_and_start()
            streamers.append(s)

    num_streamers = len(streamers)

    # --- Build SparseInterconnectArbiter ---
    # This replaces the old streamer_to_port / bank_rr_priority
    # with a cycle-accurate model of the RTL SparseInterconnect,
    # including per-bank RoundRobinArbiter state.
    tcdm_arbiter = build_sparse_interconnect(
        streamers, rw_pairs, num_banks,
        sparse_port_config=sparse_port_config,
        dormant_ports=dormant_ports,
    )

    # Two-stage response pipeline modeling TCDM + register latency:
    # Cycle N:   request granted (Phase 3) → next_pending_responses
    # Cycle N+1: response exits TCDM shift_reg (pending_responses)
    # Cycle N+2: data registered into data_buffer (pending_buffer_writes)
    pending_responses: list[tuple[int, int]] = []
    pending_buffer_writes: list[tuple[int, int]] = []

    # --- Build BlockGemmState accelerator model ---
    # Identify streamer indices for a, b, c, d and determine K / M*N.
    # Convention: readers come first (A, B), then the RW split (C reader,
    # C writer).  The K dimension is the one C is invariant to.
    a_idx = b_idx = c_idx = d_idx = -1
    reader_count = 0
    for si, s in enumerate(streamers):
        if s.is_reader:
            if reader_count == 0:
                a_idx = si
            elif reader_count == 1:
                b_idx = si
            else:
                # Third reader = C reader (from RW split)
                c_idx = si
            reader_count += 1
        elif s.is_writer:
            d_idx = si

    # Determine K from tiling: product of bounds for dimensions
    # that C (RW operand) is invariant to.
    # The RW operand's invariant_dims tells us which dims are K.
    rw_desc = None
    for desc in operand_descriptors:
        if desc.kind == OperandKind.READER_WRITER:
            rw_desc = desc
            break
    k_dims = rw_desc.invariant_dims if rw_desc else frozenset()
    K_val = 1
    M_N_val = 1
    found_inner_loop = False
    for dim_idx, tile_size, _ in tiling:
        if dim_idx in k_dims and not found_inner_loop:
            K_val *= tile_size
        else:
            M_N_val *= tile_size
        if tile_size > 1:
            found_inner_loop = True

    acc = BlockGemmState(
        K=K_val,
        M_N=M_N_val,
        a_streamer_idx=a_idx,
        b_streamer_idx=b_idx,
        c_streamer_idx=c_idx,
        d_streamer_idx=d_idx,
    )

    cycle = 0
    MAX_CYCLES = total_steps * num_streamers * 2

    while cycle < MAX_CYCLES:
        # --- Termination check ---
        if acc.is_done and not pending_responses and not pending_buffer_writes and all(
            s.is_fully_done for s in streamers
        ):
            break

        cycle += 1
    
        if cycle == 1:
            acc.busy = True  # For debugging: force busy to test preconditions

        # ==============================================================
        # Phase 0: Deliver responses & advance pipeline
        # ==============================================================
        # Readers: 2-cycle latency — grant (N) → shift_reg (N+1) →
        #   data_buffer register (N+2).
        # Writers: data leaves with the request at grant time (Phase 3).
        #   Per-channel data buffers are popped in grant_channel().
        #   The response pipeline is reader-only.
        next_pending_responses: list[tuple[int, int]] = []

        # Snapshot writer buffer fullness BEFORE any grants (pipe=false).
        # The ComplexQueueConcat input fires only when ALL per-channel
        # queues have room.  With pipe=false, even if a grant frees a
        # slot later this cycle, the input side still sees "full".
        for s in streamers:
            if s.is_writer:
                s._writer_buf_full_at_start = any(
                    len(buf) >= s.data_buffer_depth
                    for buf in s.writer_channel_bufs
                )
                # Reset writerPort activity from last cycle
                s._writer_port_cache_index = None

        # Stage 2 (readers only): register TCDM read data into dataBuffer
        for si, step in pending_buffer_writes:
            s = streamers[si]
            if s.is_reader:
                s.data_buffer.append(step)

        # Stage 1 (readers only): promote shift_reg outputs to stage 2
        next_buffer_writes: list[tuple[int, int]] = []
        for si, step in pending_responses:
            s = streamers[si]
            if s.is_reader:
                next_buffer_writes.append((si, step))
        pending_buffer_writes = next_buffer_writes
        pending_responses = []

        # ==============================================================
        # Phase 1: AGU — generate addresses into outputBuffer
        # ==============================================================
        # Each AGU ticks at most once per cycle.  The counter only
        # advances when ALL per-channel output buffer queues accept
        # (mirrors counters_tick = sBUSY && outputBuffer.in.head.fire
        # where fire requires enq_all_ready across all channel queues).
        if cycle == 6:
            pass #For debugging: keep this here dont remove

        for s in streamers:
            if s.agu_tick_possible():
                s.do_agu_tick()

        # ==============================================================
        # Phase 2: Per-channel request issue
        # ==============================================================
        # Each channel independently issues a TCDM request when it
        # has an address in its output buffer queue and meets
        # streamer-level preconditions.
        #
        # The RTL DataResponser has a combinational bypass:
        #   rspReady = ~lastVal || dataFifoPopped
        # So even when responser_slots == 0, a reader can issue if
        # its dataBuffer will be popped this cycle.  Pre-compute
        # this per-reader before issuing requests.
        #
        # dataFifoPopped sources:
        # (a) Non-cache reader: accelerator fire → reader_consume
        # (b) Cache reader: cache_process consuming updateCache from
        #     dataBuffer (independent of acc fire for the pop itself,
        #     but canAcceptNew depends on acc consuming cache output).

        # Compute per-reader dataFifoPopped bypass using BlockGemmState
        data_fifo_popped: dict[int, bool] = {}

        if cycle == 6:
            pass #For debugging: keep this here dont remove

        for si, s in enumerate(streamers):
            if not s.is_reader:
                continue
            data_fifo_popped[si] = acc.predict_data_fifo_popped(si, streamers)

        # For RW pairs: writer side has priority.  If the writer has
        # any pending or requestable channels, the reader is blocked.
        rw_blocked: set[int] = set()

        for pair in rw_pairs:
            ws = streamers[pair.writer_idx]
            writer_wants = ws.has_pending_channels() or any(
                ws.channel_can_request(ch) for ch in range(ws.spatial_banks)
            )
            if writer_wants:
                rw_blocked.add(pair.reader_idx)
                pair.last_sel_writer = True
            else:
                pair.last_sel_writer = False

        if cycle == 13:
            pass #For debugging: keep this here dont remove

        for si, s in enumerate(streamers):
            if si in rw_blocked:
                continue
            bypass = data_fifo_popped.get(si, False)
            for ch in range(s.spatial_banks):
                if s.channel_can_request(ch, data_fifo_popped=bypass):
                    s.start_channel_request(ch)

        # ==============================================================
        # Phase 3: Bank arbitration — SparseInterconnect model
        # ==============================================================
        # Cycle-accurate per-bank round-robin arbitration matching the
        # RTL SparseInterconnect → ArbitrationTree → RoundRobinArbiter
        # pipeline.  Each bank selects at most one winner per cycle.
        if cycle == 23:
            pass #For debugging: keep this here dont remove

        bank_to_requestors: dict[int, list[tuple[int, int]]] = {}
        for si, s in enumerate(streamers):
            if si in rw_blocked:
                continue
            for ch, bank in enumerate(s.channel_pending_bank):
                if bank is not None:
                    bank_to_requestors.setdefault(bank, []).append((si, ch))

        granted = tcdm_arbiter.arbitrate(bank_to_requestors)
        for si, ch, bank in granted:
            completed = streamers[si].grant_channel(ch)
            # Only readers need the response pipeline (data arrives later).
            # Writers already consumed their data at grant time.
            if streamers[si].is_reader:
                for step in completed:
                    next_pending_responses.append((si, step))

        # Store completed-step responses for delivery next cycle
        pending_responses = next_pending_responses

        # ==============================================================
        # Phase 4: Accelerator tick (BlockGemmState)
        # ==============================================================
        # The BlockGemmState models the full accelerator state machine
        # with independent per-streamer ready/valid signals, register
        # cut pipeline, K-accumulation, and d_o back-pressure.
        if cycle == 28:
            pass #For debugging: keep this here dont remove


        acc.tick(streamers)

        if acc.compute_fired:
            cycle_for_step_i[acc.acc_step - 1] += 1

        for si, s in enumerate(streamers):
            s.old_cib = s.cache_instruction_buffer.copy()

        # ==============================================================
        # Phase 5: Reader consumption (all readers)
        # ==============================================================
        # tick() only sets consumed flags; the actual buffer mutations
        # (dataBuffer pops, cache_process) happen here so that ALL
        # readers are handled uniformly via the consumed flags.
        for si, s in enumerate(streamers):
            if not s.is_reader:
                continue
            # Look up the per-streamer consumed flag
            if si == acc.a_streamer_idx:
                consumed = acc.a_consumed
                consumed_agu_step = acc.a_consumed_agu_step
            elif si == acc.b_streamer_idx:
                consumed = acc.b_consumed
                consumed_agu_step = acc.b_consumed_agu_step
            elif si == acc.c_streamer_idx:
                consumed = acc.c_consumed
                consumed_agu_step = acc.c_consumed_agu_step
            else:
                consumed = False
                consumed_agu_step = -1

            if s.uses_fixed_cache:
                # Cache reader: process FixedLevelCache + cache counter
                # ticks.  For new reader model this also handles
                # dual-bank pipeline and independent counter ticks.
                # For old writer model this handles the delayedValid
                # register.
                # For from_rw readers, pass the paired writer's
                # writerPort cache index for dual-port bank conflict.
                wpc_idx = None
                if s.from_rw and s._paired_writer_idx is not None:
                    wpc_idx = streamers[s._paired_writer_idx]._writer_port_cache_index
                s.cache_process(consumed, writer_port_cache_index=wpc_idx)
            elif consumed and s.needs_tcdm_access(consumed_agu_step):
                # Non-cache reader: pop the consumed entry from dataBuffer.
                # Guard: invariant steps have no buffer entry to pop.
                s.reader_consume(consumed_agu_step)

    if cycle >= MAX_CYCLES:
        return -1
    return cycle


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

def get_cost_model(name: str = "hardware_latency") -> CostModel:
    """Return a cost model instance by name."""
    models: dict[str, CostModel] = {
        "energy": EnergyCostModel(),
        "latency": LatencyCostModel(),
        "hardware_latency": HardwareLatencyCostModel(),
    }
    if name not in models:
        raise ValueError(f"Unknown cost model '{name}'. Choose from {list(models)}")
    return models[name]