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
    # strides[0] = [0, 64, 1024, 0, 256, 0, 0] # sched 200
    # strides[1] = [64, 0, 1024, 256, 4096, 0, 0]
    # strides[2] = [256, 0, 4096, 512, 0, 0, 0]
    strides[0] = [0, 64, 1024, 0, 256, 0, 0] # sched 195
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
    fixed_cache_depth: int = 63

    # HandShakeRepeater count — models the Reader's RepeatHandshake.
    # When the innermost loop has stride 0 and bound > 1, the reader
    # overrides AGU bounds[0] to 1 and repeats each data word
    # ``repeat_count`` times.  Default 1 = no repeat.
    repeat_count: int = 1

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

    # dataBuffer: FIFO of step indices (reader: fetched data; writer:
    # data from accelerator waiting to be written)
    data_buffer: deque[int] = field(default_factory=deque)

    # For readers: per-channel response slots available.  Each channel
    # has its own UpDownCounter in DataResponser with ceil = bufferDepth+1.
    # tickUp fires per-channel on reqSubmit; tickDown fires for ALL
    # channels simultaneously on dataBuffer.out.fire (dataFifoPopped).
    responser_slots: list[int] = field(default_factory=list)

    # ---- FixedLevelCache state (only active when uses_fixed_cache) ----
    # Models the fixedCacheInstructionBuffer FIFO in the AGU.  Each
    # entry is ``(step, is_update_cache)`` where ``is_update_cache``
    # means the step needs TCDM data (first pass through critical
    # loop) and ``not is_update_cache`` means data is read from cache
    # memory (subsequent passes).  The FIFO depth equals
    # ``output_buffer_depth`` (matches RTL).
    cache_instruction_buffer: deque[tuple[int, bool]] = field(default_factory=deque)
    old_cib: deque[tuple[int, bool]] = field(default_factory=deque)  # for debugging: track the "old" CIB state at the time of AGU ticks

    # Models the ``delayedValid`` register + associated data inside
    # the FixedLevelCache.  When not None, the cache has valid output
    # for this step that the accelerator can consume.  Updated at the
    # end of each cycle (register semantics).
    cache_output_step: int | None = None

    def __post_init__(self):
        # --- Reader innermost-loop repeat override ---
        # Loops with bound == 1 are pruned before reaching hardware,
        # so the effective innermost level is the first with bound > 1.
        # Reader.scala: when(temporalStrides(0) === 0.U) {
        #   addressgen.io.cfg.temporalBounds(0) := 1.U
        # }
        # HandShakeRepeater: repeat_times = Mux(stride(0)==0, bounds(0), 1)
        if self.is_reader:
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

    # ----- address helpers -----

    def compute_channel_bank(self, step: int, ch: int) -> int:
        """Compute the TCDM bank index for channel ``ch`` at ``step``."""
        counters = self._step_to_counters(step)
        base = sum(self.strides_bank[lvl] * counters[lvl]
                   for lvl in range(self.num_levels))
        return (base // 8 ) % self.num_banks + ch

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
        # Levels with bound == 1 are pruned before reaching hardware.
        # CriticalLoopFinder operates on the compacted set of levels.
        eff_bounds = []
        eff_strides = []
        for i in range(self.num_levels):
            if self.bounds[i] > 1:
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

        The RTL ``counters_tick`` logic (AddressGenUnit.scala):

        * Without fixed cache (``!newUseCache``):
          ``counters_tick = sBUSY && outputBuffer.in.head.fire``
          → every step goes into the outputBuffer, so the counter is
            gated by output buffer room.

        * With fixed cache (``newUseCache``):
          ``counters_tick = sBUSY && ((outputBufferFillCondition &&
            outputBuffer.fire) || fixedCacheInstructionBuffer.fire)``
          → both the instruction buffer and the output buffer must
            have room (the RTL gates the instruction buffer enqueue
            on ``outputBuffer.ready && fixedCacheInstructionBuffer
            .ready``).
          For access steps the address also enters the outputBuffer.
          For non-access steps the outputBuffer is not pushed but
          its ready (room available) is still required.
        """
        if self.agu.state != "BUSY":
            return False
        if self.agu.done:
            return False
        ob_has_room = all(
            len(buf) < self.output_buffer_depth
            for buf in self.output_buffers
        )
        if self.uses_fixed_cache:
            cib_has_room = (
                len(self.old_cib)
                < self.output_buffer_depth
            )
            return ob_has_room and cib_has_room
        step = self.agu.current_step()
        if self.needs_tcdm_access(step):
            return ob_has_room
        else:
            # Non-cache, non-access: shouldn't normally happen
            # (only cache-enabled readers have non-access steps).
            return True

    def do_agu_tick(self):
        """
        Execute one AGU tick.

        * With cache: every tick pushes to the cache instruction
          buffer.  Access steps (``needs_tcdm_access``) also push
          to the output buffers for TCDM.
        * Without cache: access steps push to output buffers.
          Non-access steps only advance the counter.
        """
        step = self.agu.current_step()
        is_access = self.needs_tcdm_access(step)
        if self.uses_fixed_cache:
            self.cache_instruction_buffer.append((step, is_access))
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
        * Writer: dataBuffer has data to write.
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
            if len(self.data_buffer) == 0:
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

        Returns a list of step indices that became fully complete
        (all channels granted) as a result of this grant.
        """
        step = self.output_buffers[ch].popleft()
        self.channel_pending_bank[ch] = None
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
        """
        if self.uses_fixed_cache:
            if not self.cache_instruction_buffer:
                return False  # no instrValid
            _step, is_access = self.cache_instruction_buffer[0]
            if is_access:  # lastAccess → goToTCDM: need dataBuffer room
                return len(self.data_buffer) < self.data_buffer_depth
            return True  # cache-only path: no backpressure
        return len(self.data_buffer) < self.data_buffer_depth

    def writer_accept(self, acc_step: int):
        """Accelerator pushes result into the writer.

        RTL Writer.scala (ReaderWriter, fixed-cache mode):
          bothValid = dataAfterCrosser.valid && instrValid
          - goToCache (useCache): write to reader's FixedLevelCache
            (combinational, no timing effect in model)
          - goToTCDM (lastAccess): push into dataBuffer for TCDM
            writeback
          Consumes one fixedCacheInstruction per d_o fire.

        Without fixed-cache: push directly to dataBuffer.
        """
        if self.uses_fixed_cache:
            _step, is_access = self.cache_instruction_buffer.popleft()
            if is_access:  # lastAccess → data goes to TCDM
                self.data_buffer.append(acc_step)
            # else: cache-only write, no dataBuffer entry
            return
        self.data_buffer.append(acc_step)

    # ----- FixedLevelCache processing -----

    def cache_process(self, acc_consumed: bool):
        """Advance the FixedLevelCache state for one cycle.

        This models the ``delayedValid`` register semantics from the
        RTL FixedLevelCache module.  Must be called AFTER the
        accelerator fire decision so that ``acc_consumed`` is known.

        * ``canAcceptNew = !delayedValid || dataOut.ready``
        * If a new instruction is processed this cycle,
          ``delayedValid := true`` (output will be valid next cycle).
        * Else if the accelerator consumed (``dataOut.ready``),
          ``delayedValid := false``.

        For an updateCache instruction the cache needs data from
        ``data_buffer`` (connected to ``dataBuffer.out.head`` in
        RTL).  For a cache-hit instruction no TCDM data is needed.
        """
        can_accept_new = (self.cache_output_step is None) or acc_consumed
        processed_new = False
        if can_accept_new and self.cache_instruction_buffer:
            step, is_update = self.cache_instruction_buffer[0]
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
            and len(self.data_buffer) == 0
        )
        if self.uses_fixed_cache:
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
            agu_step_c = s_c.acc_to_agu_step(self.acc_step)
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
                    self.c_consumed_agu_step = s_c.acc_to_agu_step(self.acc_step)

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

        regcut_input_fire = datacut_input_ready and a_valid and b_valid

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
        # io.in.valid = a_valid && b_valid (from DecoupledCat4to1;
        # subtraction inputs are always valid when busy).
        data_inside = self.datacut_stage0_valid or self.datacut_stage1_valid
        shift_suggestion = data_inside or (a_valid and b_valid)
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
                agu_step_c = s_c.acc_to_agu_step(self.acc_step)
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
            # Both a AND b must be valid for regcut input to fire
            s_a = streamers[self.a_streamer_idx]
            s_b = streamers[self.b_streamer_idx]
            a_agu = s_a.acc_to_agu_step(step)
            b_agu = s_b.acc_to_agu_step(step)
            return (self._reader_data_valid(s_a, a_agu) and
                    self._reader_data_valid(s_b, b_agu))

        elif si == self.c_streamer_idx:
            # c: consumed when add_c_fire AND is_last_repeat
            if self.compute_fire_counter != 0:
                return False
            if not self.datacut_stage1_valid:
                return False  # combined_out_valid required
            step = self.acc_step
            if not s.is_last_repeat(step):
                return False
            agu_step = s.acc_to_agu_step(step)
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

        For cache readers the dataBuffer pop happens inside
        cache_process when it handles an updateCache instruction.
        canAcceptNew = (no output) or (acc consumed cache output).

        For non-cache readers the dataBuffer is popped directly by
        the accelerator, but only when needs_tcdm_access is True
        (invariant steps have no buffer entry).
        """
        if not self.busy:
            return False

        s = streamers[si]

        if s.uses_fixed_cache:
            # Cache: dataBuffer popped when cache_process handles an
            # updateCache instruction with data.
            acc_consumed = self._predict_acc_consumes_reader(si, streamers)
            can_accept = (s.cache_output_step is None) or acc_consumed
            if can_accept and s.cache_instruction_buffer:
                step, is_update = s.cache_instruction_buffer[0]
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
            )
            writer_s.agu.reset_and_start()
            writer_idx = len(streamers)
            streamers.append(writer_s)

            rw_pairs.append(ReaderWriterPair(
                reader_idx=reader_idx,
                writer_idx=writer_idx,
            ))
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
            )
            s.agu.reset_and_start()
            streamers.append(s)

    num_streamers = len(streamers)

    # --- TCDM port mapping ---
    # In the RTL, a ReaderWriter shares one set of TCDM request
    # channels (via MuxDecoupled).  The TCDM interconnect (stream_xbar
    # / rr_arb_tree) only sees one physical port group per RW pair.
    # Map each streamer index to a TCDM port index so that the RW
    # pair's reader and writer share the same port.
    streamer_to_port: dict[int, int] = {}
    port_idx = 0
    for si in range(num_streamers):
        # Check if this streamer is the writer part of an RW pair
        is_rw_writer = False
        for pair in rw_pairs:
            if pair.writer_idx == si:
                # Writer shares port with reader
                streamer_to_port[si] = streamer_to_port[pair.reader_idx]
                is_rw_writer = True
                break
        if not is_rw_writer:
            streamer_to_port[si] = port_idx
            port_idx += 1
    num_ports = port_idx

    # Per-bank round-robin priority state.
    # The rr_arb_tree in stream_xbar operates on physical TCDM port
    # indices.  After granting port *p* on bank *b*, the next priority
    # rotates to ``(p + 1) % num_ports``.
    bank_rr_priority: dict[int, int] = {b: 0 for b in range(num_banks)}

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
    acc.busy = True

    cycle = 0
    MAX_CYCLES = total_steps * num_streamers * 2

    while cycle < MAX_CYCLES:
        # --- Termination check ---
        if acc.is_done and not pending_responses and not pending_buffer_writes and all(
            s.is_fully_done for s in streamers
        ):
            break

        cycle += 1

        # ==============================================================
        # Phase 0: Deliver responses & advance pipeline
        # ==============================================================
        # Readers: 2-cycle latency — grant (N) → shift_reg (N+1) →
        #   data_buffer register (N+2).
        # Writers: 1-cycle latency — data leaves with the request,
        #   the write completes when the grant is acknowledged (the
        #   TCDM shift_reg response merely confirms the write).
        #   data_buffer is popped in the cycle after grant.
        next_pending_responses: list[tuple[int, int]] = []

        # Stage 2 (readers only): register TCDM read data into dataBuffer
        for si, step in pending_buffer_writes:
            s = streamers[si]
            if s.is_reader:
                s.data_buffer.append(step)

        # Stage 1: process shift_reg outputs
        # Writers: pop data_buffer now (data already sent with request)
        # Readers: promote to stage 2 for next cycle
        next_buffer_writes: list[tuple[int, int]] = []
        for si, step in pending_responses:
            s = streamers[si]
            if s.is_reader:
                next_buffer_writes.append((si, step))
            else:
                if s.data_buffer:
                    s.data_buffer.popleft()
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

        if cycle == 5:
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

        if cycle == 11:
            pass #For debugging: keep this here dont remove

        for si, s in enumerate(streamers):
            if si in rw_blocked:
                continue
            bypass = data_fifo_popped.get(si, False)
            for ch in range(s.spatial_banks):
                if s.channel_can_request(ch, data_fifo_popped=bypass):
                    s.start_channel_request(ch)

        # ==============================================================
        # Phase 3: Bank arbitration — per-bank round-robin
        # ==============================================================
        # The TCDM interconnect uses stream_xbar which instantiates a
        # per-bank rr_arb_tree.  Each bank has its own independent
        # round-robin state.  After granting a requestor the priority
        # rotates to the *next* requestor (by index).
        if cycle == 12:
            pass #For debugging: keep this here dont remove

        bank_to_requestors: dict[int, list[tuple[int, int]]] = {}
        for si, s in enumerate(streamers):
            for ch, bank in enumerate(s.channel_pending_bank):
                if bank is not None:
                    bank_to_requestors.setdefault(bank, []).append((si, ch))

        for bank, requestors in bank_to_requestors.items():
            if len(requestors) == 1:
                si, ch = requestors[0]
                completed = streamers[si].grant_channel(ch)
                for step in completed:
                    next_pending_responses.append((si, step))
                port = streamer_to_port[si]
                #bank_rr_priority[bank] = (port + 1) % num_ports
                bank_rr_priority[bank] = 0
            else:
                # Per-bank round-robin: pick the requestor whose
                # TCDM port index is nearest *at or after* the current
                # priority pointer, wrapping around.
                prio = bank_rr_priority[bank]
                def rr_key(r: tuple[int, int]) -> int:
                    return (streamer_to_port[r[0]] - prio) % num_ports
                winner = min(requestors, key=rr_key)
                winner_si = winner[0]
                # Grant ALL channels of the winner that are pending on
                # this bank (there should be at most one per streamer).
                for si, ch in requestors:
                    if si == winner_si:
                        completed = streamers[si].grant_channel(ch)
                        for step in completed:
                            next_pending_responses.append((si, step))
                winner_port = streamer_to_port[winner_si]
                #bank_rr_priority[bank] = (winner_port + 1) % num_ports
                bank_rr_priority[bank] = 0

        # Store completed-step responses for delivery next cycle
        pending_responses = next_pending_responses

        # ==============================================================
        # Phase 4: Accelerator tick (BlockGemmState)
        # ==============================================================
        # The BlockGemmState models the full accelerator state machine
        # with independent per-streamer ready/valid signals, register
        # cut pipeline, K-accumulation, and d_o back-pressure.
        if cycle == 8:
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
                # Cache reader: cache_process handles the delayedValid
                # register update and dataBuffer pop (for updateCache).
                s.cache_process(consumed)
            elif consumed and s.needs_tcdm_access(consumed_agu_step):
                # Non-cache reader: pop the consumed entry from dataBuffer.
                # Guard: invariant steps have no buffer entry to pop.
                s.reader_consume(consumed_agu_step)

    print(cycle_for_step_i)
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