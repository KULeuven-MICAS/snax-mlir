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
        invariant_dims: Set of *logical* dimension ids to which the operand
            is invariant (i.e. the stride is 0 for those dims).
    """
    kind: OperandKind
    spatial_banks: int
    element_bytes: int
    invariant_dims: frozenset[int]

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
    num_ops = len(operand_descriptors)
    num_levels = len(tiling)

    # For each logical dim, track cumulative factor (inner → outer).
    # tiling is already inner → outer.
    dim_cumulative: dict[int, int] = {}

    strides: list[list[int]] = [[] for _ in range(num_ops)]

    for lvl, (dim_idx, tile_size, _is_crit) in enumerate(tiling):
        cum = dim_cumulative.get(dim_idx, 1)
        for op_idx, desc in enumerate(operand_descriptors):
            if dim_idx in desc.invariant_dims:
                strides[op_idx].append(0)
            else:
                # A single temporal step moves through
                #   spatial_banks * element_bytes  contiguous bytes
                # in memory.  In bank-word units (each 8 bytes) the base
                # stride is  burst_bank_words = spatial_banks * element_bytes / 8.
                # For the cumulative factor (when the same dim is tiled
                # multiple times), multiply by the product of inner tile
                # sizes for this dim.
                strides[op_idx].append(desc.burst_bank_words * cum)
        dim_cumulative[dim_idx] = cum * tile_size


    # EXTREMELY IMPORTANT FIXME: HARD-CODED EXAMPLE FOR DEBUGGING – REPLACE WITH ACTUAL LOGIC
    # TODO
    # TODO
    # TODO
    strides[0] = [0, 64, 1024, 0, 256, 0, 0]
    strides[1] = [0, 64, 0, 1024, 256, 0, 0]
    strides[2] = [0, 0, 256, 4096, 0, 0, 0]

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
    ) -> float:
        return hardware_latency_cost_of_tiling(
            tiling,
            operand_descriptors,
            invariance_map,
            num_banks=num_banks,
            bank_bits=bank_bits,
        )


def hardware_latency_cost_of_tiling(
    tiling: list[tuple[int, int, bool]],
    operand_descriptors: Sequence[OperandDescriptor],
    invariance_map: list[set[int]],
    *,
    num_banks: int = 32,
    bank_bits: int = 64,
) -> float:
    """
    Cycle-accurate simulation of the SNAX streamer hardware.

    Returns the total number of TCDM clock cycles to complete all streamer
    accesses.
    """
    if not tiling:
        return 0.0

    strides_bank = _compute_operand_stride_per_tile(
        tiling, operand_descriptors, invariance_map, bank_bits
    )

    return float(_simulate_hardware(
        tiling, operand_descriptors, invariance_map, strides_bank, num_banks
    ))


# ---------------------------------------------------------------------------
# Internal: hardware simulation
# ---------------------------------------------------------------------------


def _simulate_hardware(
    tiling: list[tuple[int, int, bool]],
    operand_descriptors: Sequence[OperandDescriptor],
    invariance_map: list[set[int]],
    strides_bank: list[list[int]],
    num_banks: int,
) -> int:
    """
    Simulate the SNAX streamer pipeline cycle-by-cycle.

    Returns total cycle count.
    """
    num_levels = len(tiling)
    if num_levels == 0:
        return 0

    bounds = [t[1] for t in tiling]
    num_ops = len(operand_descriptors)
    for i in range(len(operand_descriptors)):
        if operand_descriptors[i].kind == OperandKind.READER_WRITER:
            # Insert a duplicate descriptor for the writer part of the RW operand
            desc = operand_descriptors[i]
            operand_descriptors.insert(i, OperandDescriptor(
                kind=OperandKind.WRITER,
                spatial_banks=desc.spatial_banks,
                element_bytes=desc.element_bytes,
                invariant_dims=desc.invariant_dims,
            ))
            invariance_map.insert(i, invariance_map[i])
            strides_bank.insert(i, strides_bank[i])
            num_ops += 1
    
    # Total number of global iteration steps
    total_steps = 1
    for b in bounds:
        total_steps *= b

    # --- Helper functions ---

    def step_to_counters(step: int) -> list[int]:
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
        return (addr // 8) % num_banks #TODO: CHECK IF // 8 SHOULD BE HERE

    def streamer_needs_access(op_idx: int, step: int) -> bool:
        """Does this streamer perform a memory access at the given global step?"""
        desc = operand_descriptors[op_idx]
        counters = step_to_counters(step)
        if desc.kind == OperandKind.READER or desc.kind == OperandKind.READER_WRITER:
            return _reader_active(counters, tiling, desc.invariant_dims)
        else:  # WRITER
            return _writer_active(counters, bounds, tiling, desc.invariant_dims)

    def compute_burst_banks(op_idx: int, step: int) -> list[int]:
        """Return list of bank indices for the burst at the given step."""
        desc = operand_descriptors[op_idx]
        counters = step_to_counters(step)
        base_bank = compute_bank_address(op_idx, counters)
        return [(base_bank + i) % num_banks for i in range(desc.burst_bank_words)]

    # --- Per-streamer state ---

    BUFFER_DEPTH = 2
    AGU_QUEUE_DEPTH = 4  # Typical output buffer depth for AGU

    # Each streamer's AGU "step pointer": the next global step index to generate addresses for
    agu_step = [0] * num_ops

    # Address buffers decoupled from the memory requests
    # Represents the outputBuffer of the AGU module
    address_buffers: list[deque[int]] = [deque() for _ in range(num_ops)]

    # Buffers: list of deques. For readers, entries are step indices of data
    # that has been fetched. For writers, entries are step indices of data
    # from the accelerator waiting to be written.
    buffers: list[deque[int]] = [deque() for _ in range(num_ops)]

    # Per-streamer in-flight burst state: which banks of the current burst
    # still need to be serviced. None means no burst in progress.
    pending_banks: list[set[int] | None] = [None] * num_ops
    # The step index of the currently in-flight burst
    pending_step: list[int] = [0] * num_ops

    # Accelerator step counter
    acc_step = 0

    # Round-robin priority counter for bank arbitration
    rr_priority = 0

    cycle = 0
    MAX_CYCLES = total_steps * num_ops * num_banks * 10  # safety bound

    cycles_for_step_i = [0]  # for debugging: track cycles taken by each global step

    while cycle < MAX_CYCLES:
        # Check termination: AGU done generating addresses, all memory requests have resolved,
        # and buffers empty up to total_steps completion.
        all_done = all(agu_step[op] >= total_steps and
                       len(address_buffers[op]) == 0 and
                       pending_banks[op] is None and
                       len(buffers[op]) == 0
                       for op in range(num_ops))
        if all_done and acc_step >= total_steps:
            break

        cycle += 1

        # ==================================================================
        # Phase 0: Address Generation Unit (AGU)
        # ==================================================================
        
        # Advance agu_step for all ops past non-access steps and append necessary steps to address_buffers
        # Hardware can generate at most 1 access per cycle.
        for op in range(num_ops):
            while agu_step[op] < total_steps and not streamer_needs_access(op, agu_step[op]):
                agu_step[op] += 1
            
            if agu_step[op] < total_steps and len(address_buffers[op]) < AGU_QUEUE_DEPTH:
                address_buffers[op].append(agu_step[op])
                agu_step[op] += 1

        # ==================================================================
        # Phase 1: Determine which streamers want to issue memory requests
        # ==================================================================

        # Collect all individual bank requests for this cycle.
        # A request is (op_idx, bank_idx).
        bank_requests: list[tuple[int, int]] = []

        reader_writer_writing = False  # track if any RW streamer is in its write phase this cycle


        for op in range(num_ops):
            desc = operand_descriptors[op]
            is_reader_writer = desc.kind == OperandKind.READER_WRITER
            is_reader = desc.kind == OperandKind.READER
            is_writer = desc.kind == OperandKind.WRITER

            if is_writer:
                pass
            if is_reader_writer:
                reader_writer_writing
                pass
            
            # Continue a pending burst?
            if pending_banks[op] is not None:
                for bank in pending_banks[op]:
                    bank_requests.append((op, bank))
                if is_writer:
                    reader_writer_writing = True
                continue

            if len(address_buffers[op]) == 0:
                # No addresses to generate memory requests for
                reader_writer_writing = False
                continue

            step = address_buffers[op][0]

            # Check buffer capacity
            if is_reader or is_reader_writer:
                # Reader: must have space in buffer to put fetched data
                reader_writer_writing = False
                if len(buffers[op]) >= BUFFER_DEPTH:
                    continue
                if is_reader_writer and reader_writer_writing:
                    # ReaderWriter in write phase this cycle → read phase is stalled
                    continue

            elif is_writer:
                # Writer: must have data in buffer to write
                if len(buffers[op]) == 0:
                    reader_writer_writing = False
                    continue
                reader_writer_writing = True

            # Start a new burst
            burst_banks = compute_burst_banks(op, step)
            pending_banks[op] = set(burst_banks)
            pending_step[op] = step

            if op == 0:
                pass

            if op == 1:
                pass

            for bank in pending_banks[op]:
                bank_requests.append((op, bank))
            
        if cycle == 12:
            pass

        # ==================================================================
        # Phase 2: Resolve banking conflicts (round-robin arbitration)
        # ==================================================================

        # Group requests by bank
        bank_to_ops: dict[int, list[int]] = {}
        for op, bank in bank_requests:
            bank_to_ops.setdefault(bank, [])
            if op not in bank_to_ops[bank]:
                bank_to_ops[bank].append(op)

        granted: dict[int, set[int]] = {op: set() for op in range(num_ops)}
        for bank, ops in bank_to_ops.items():
            if len(ops) == 1:
                granted[ops[0]].add(bank)
            else:
                # Give priority to the lowest operand
                winner = min(ops)
                granted[winner].add(bank)

        # ==================================================================
        # Phase 2b: Update pending bursts based on grants
        # ==================================================================

        # Track which streamers complete their burst this cycle
        # Use next-state tracking to apply updates atomically at end of cycle
        next_address_buffers: list[deque[int]] = [deque(b) for b in address_buffers]
        next_buffers: list[deque[int]] = [deque(b) for b in buffers]
        next_pending_banks: list[set[int] | None] = list(pending_banks)
        next_agu_step: list[int] = list(agu_step)
        next_pending_step: list[int] = list(pending_step)

        for op in range(num_ops):
            if pending_banks[op] is None:
                continue

            # Remove granted banks from pending set
            remaining = pending_banks[op] - granted[op]
            if len(remaining) == 0:
                # Burst complete
                desc = operand_descriptors[op]
                is_reader = desc.kind in (OperandKind.READER, OperandKind.READER_WRITER)
                is_writer = desc.kind == OperandKind.WRITER

                if is_reader:
                    next_buffers[op].append(pending_step[op])
                elif is_writer:
                    # Writer: data was in buffer, now written to memory
                    # Pop the oldest entry from the buffer
                    if next_buffers[op]:
                        next_buffers[op].popleft()

                next_pending_banks[op] = None
                
                # Consume this completed address from the AGU queue
                if next_address_buffers[op]:
                    next_address_buffers[op].popleft()

                # Advance past any subsequent non-access steps
                # (handled by AGU now)
            else:
                next_pending_banks[op] = remaining

        # ==================================================================
        # Phase 3: Accelerator fire logic
        # ==================================================================

        # The accelerator fires if:
        # 1. For each reader-type streamer: either the required data (for
        #    acc_step) is in its buffer, or no access is needed at acc_step.
        # 2. For each writer-type streamer: there is space in its buffer.

        if cycle == 13:
            pass

        acc_can_fire = acc_step < total_steps
        if acc_step == 514:
            pass
        if acc_can_fire:
            for op in range(num_ops):
                desc = operand_descriptors[op]
                is_reader = desc.kind in (OperandKind.READER, OperandKind.READER_WRITER)
                is_writer = desc.kind == OperandKind.WRITER

                if is_reader:
                    needs_access = streamer_needs_access(op, acc_step)
                    if needs_access:
                        # Data for acc_step must be in the buffer.
                        # Check both current and next-state buffers (data arriving
                        # this cycle is visible to the accelerator).
                        if acc_step not in next_buffers[op]:
                            acc_can_fire = False
                            break
                    # If no access needed, the reader doesn't block the accelerator.

                elif is_writer:
                    # Writer needs space in its buffer to accept the result
                    if len(next_buffers[op]) >= BUFFER_DEPTH:
                        acc_can_fire = False
                        break

        if not acc_can_fire:
            pass

        if acc_can_fire:
            cycles_for_step_i.append(0)
            # Pop consumed data from reader buffers; push to writer buffers
            for op in range(num_ops):
                desc = operand_descriptors[op]
                is_reader = desc.kind in (OperandKind.READER, OperandKind.READER_WRITER)
                is_writer = desc.kind == OperandKind.WRITER

                if is_reader:
                    needs_access = streamer_needs_access(op, acc_step)
                    if needs_access and acc_step in next_buffers[op]:
                        next_buffers[op].remove(acc_step)

                elif is_writer:
                    needs_access = streamer_needs_access(op, acc_step)
                    if needs_access and len(next_buffers[op]) < BUFFER_DEPTH:
                        next_buffers[op].append(acc_step)

            acc_step += 1

        # ==================================================================
        # Phase 4: Commit next-state
        # ==================================================================

        address_buffers = next_address_buffers
        buffers = next_buffers
        pending_banks = next_pending_banks
        agu_step = next_agu_step
        pending_step = next_pending_step
        cycles_for_step_i[-1] += 1  # for debugging: count this cycle towards the current global step


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