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

    def __post_init__(self):
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
        return ((base + ch) // 8) % self.num_banks

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
          → access steps need output buffer room;
            non-access steps tick via fixedCacheInstructionBuffer
            (same depth, always consumed in parallel → not a
            bottleneck).  We don't model the instruction buffer, so
            non-access steps tick freely.

        Combined: for an access step, all per-channel output queues
        must have room.  For a non-access step, the counter ticks
        unconditionally.
        """
        if self.agu.state != "BUSY":
            return False
        if self.agu.done:
            return False
        step = self.agu.current_step()
        if self.needs_tcdm_access(step):
            # Access step: outputBuffer must accept → all channel
            # queues need room (enq_all_ready in ComplexQueueConcat)
            return all(len(buf) < self.output_buffer_depth
                       for buf in self.output_buffers)
        else:
            # Non-access step: fixedCacheInstructionBuffer would
            # accept; we don't model it, so tick freely.
            return True

    def do_agu_tick(self):
        """
        Execute one AGU tick.

        * Access step (``needs_tcdm_access`` is True):
          Push the step index into ALL per-channel output buffer
          queues simultaneously (matching ``outputBufferFillCondition
          || !newUseCache`` in the RTL) and advance the counter.

        * Non-access step: only advance the counter — no entry is
          placed in the output buffer because no TCDM transaction is
          needed.  The counter still ticks (via the
          fixedCacheInstructionBuffer path in hardware).
        """
        step = self.agu.current_step()
        if self.needs_tcdm_access(step):
            for buf in self.output_buffers:
                buf.append(step)
            self.step_grants_remaining[step] = self.spatial_banks
        self.agu.tick()

    # ----- per-channel request / grant helpers -----

    def channel_can_request(self, ch: int) -> bool:
        """Can channel ``ch`` issue a new TCDM request this cycle?

        Mirrors the DataRequestor fire conditions:
        * Channel has a valid address (its output buffer queue is
          non-empty).
        * Not already mid-request (``channel_pending_bank[ch]`` is
          None).
        * Reader: DataResponser has room (``responser_slots > 0``).
        * Writer: dataBuffer has data to write.
        """
        if self.channel_pending_bank[ch] is not None:
            return False  # already has a pending request
        if not self.output_buffers[ch]:
            return False  # no address in queue
        # Streamer-level preconditions
        if self.kind in (OperandKind.READER, OperandKind.READER_WRITER):
            if self.responser_slots[ch] <= 0:
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

    def reader_data_available(self, acc_step: int) -> bool:
        """Does the dataBuffer contain the data for ``acc_step``?"""
        return acc_step in self.data_buffer

    def reader_consume(self, acc_step: int):
        """Pop ``acc_step`` from the dataBuffer (accelerator consumed it)."""
        self.data_buffer.remove(acc_step)
        # dataFifoPopped fires for ALL channels simultaneously
        for ch in range(self.spatial_banks):
            self.responser_slots[ch] += 1

    def writer_has_space(self) -> bool:
        return len(self.data_buffer) < self.data_buffer_depth

    def writer_accept(self, acc_step: int):
        """Accelerator pushes result into the writer's dataBuffer."""
        self.data_buffer.append(acc_step)

    @property
    def is_reader(self) -> bool:
        return self.kind in (OperandKind.READER, OperandKind.READER_WRITER)

    @property
    def is_writer(self) -> bool:
        return self.kind == OperandKind.WRITER

    @property
    def is_fully_done(self) -> bool:
        """AGU finished, no pending requests, buffers drained."""
        return (
            self.agu.done
            and all(len(buf) == 0 for buf in self.output_buffers)
            and not self.has_pending_channels()
            and len(self.step_grants_remaining) == 0
            and len(self.data_buffer) == 0
        )


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

    # cycle_for_step_i = [0] * total_steps  # for debugging: cycle count when accelerator step i fires

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

    # Per-bank round-robin priority state.
    # After granting streamer *w* on bank *b*, the next priority on that
    # bank rotates to the requestor index *after* w.  We store the
    # "next priority" streamer index per bank.  Requestor indices are
    # the global channel-request indices (streamer_idx * max_channels +
    # ch_idx) flattened, but since the hardware's rr_arb_tree operates
    # on the physical port indices wired into that bank, we simply
    # track the *streamer index* that has next priority for each bank.
    bank_rr_priority: dict[int, int] = {b: 0 for b in range(num_banks)}

    # Pending response queue: bursts that completed in the *previous*
    # cycle.  Due to MemoryResponseLatency = 1 in the TCDM interconnect,
    # response data arrives 1 cycle after the request is granted.
    # Entries are (streamer_index, step) tuples.
    pending_responses: list[tuple[int, int]] = []

    # Accelerator step counter
    acc_step = 0
    cycle = 0
    MAX_CYCLES = total_steps * num_streamers * num_banks * 10

    while cycle < MAX_CYCLES:
        # --- Termination check ---
        if acc_step >= total_steps and not pending_responses and all(
            s.is_fully_done for s in streamers
        ):
            break

        cycle += 1

        # ==============================================================
        # Phase 0: Deliver responses from *previous* cycle
        # ==============================================================
        # TCDM response latency = 1 cycle: data requested in cycle N-1
        # arrives in cycle N.  For readers this pushes into the
        # dataBuffer; for writers this pops the written data.
        next_pending_responses: list[tuple[int, int]] = []
        for si, step in pending_responses:
            s = streamers[si]
            if s.is_reader:
                s.data_buffer.append(step)
            else:
                # Writer: data was in buffer, now written to TCDM
                if s.data_buffer:
                    s.data_buffer.popleft()
        pending_responses = []

        # ==============================================================
        # Phase 1: AGU — generate addresses into outputBuffer
        # ==============================================================
        # Each AGU ticks at most once per cycle.  The counter only
        # advances when ALL per-channel output buffer queues accept
        # (mirrors counters_tick = sBUSY && outputBuffer.in.head.fire
        # where fire requires enq_all_ready across all channel queues).
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

        for si, s in enumerate(streamers):
            if si in rw_blocked:
                continue
            for ch in range(s.spatial_banks):
                if s.channel_can_request(ch):
                    s.start_channel_request(ch)

        # ==============================================================
        # Phase 3: Bank arbitration — per-bank round-robin
        # ==============================================================
        # The TCDM interconnect uses stream_xbar which instantiates a
        # per-bank rr_arb_tree.  Each bank has its own independent
        # round-robin state.  After granting a requestor the priority
        # rotates to the *next* requestor (by index).
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
                bank_rr_priority[bank] = (si + 1) % num_streamers
            else:
                # Per-bank round-robin: pick the requestor whose
                # streamer index is nearest *at or after* the current
                # priority pointer, wrapping around.
                prio = bank_rr_priority[bank]
                def rr_key(r: tuple[int, int]) -> int:
                    return (r[0] - prio) % num_streamers
                winner = min(requestors, key=rr_key)
                winner_si = winner[0]
                # Grant ALL channels of the winner that are pending on
                # this bank (there should be at most one per streamer).
                for si, ch in requestors:
                    if si == winner_si:
                        completed = streamers[si].grant_channel(ch)
                        for step in completed:
                            next_pending_responses.append((si, step))
                bank_rr_priority[bank] = (winner_si + 1) % num_streamers

        # Store completed-step responses for delivery next cycle
        pending_responses = next_pending_responses

        # ==============================================================
        # Phase 4: Accelerator fire logic
        # ==============================================================
        # The accelerator fires when every reader that needs a TCDM
        # access at this step has its data in the dataBuffer, and
        # every writer that needs a TCDM access has space.  Streamers
        # whose invariant-dim gating says no access is needed at
        # ``acc_step`` do not block the accelerator.
        acc_can_fire = acc_step < total_steps
        # if acc_can_fire:
        #     cycle_for_step_i[acc_step] += 1  # record when this step fired

        if acc_step == 5:
            pass #For debugging: keep this here dont remove

        if acc_can_fire:
            for s in streamers:
                if s.is_reader:
                    if s.needs_tcdm_access(acc_step):
                        if not s.reader_data_available(acc_step):
                            acc_can_fire = False
                            break
                elif s.is_writer:
                    if s.needs_tcdm_access(acc_step):
                        if not s.writer_has_space():
                            acc_can_fire = False
                            break

        if acc_can_fire:
            for s in streamers:
                if s.is_reader:
                    if s.needs_tcdm_access(acc_step):
                        s.reader_consume(acc_step)
                elif s.is_writer:
                    if s.needs_tcdm_access(acc_step):
                        s.writer_accept(acc_step)
            acc_step += 1

    # print(cycle_for_step_i)
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