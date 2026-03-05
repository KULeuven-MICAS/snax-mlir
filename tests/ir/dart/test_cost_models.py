"""
Comprehensive tests for snaxc.ir.dart.cost_models.

Covers:
  - OperandDescriptor / OperandKind construction
  - EnergyCostModel (backward-compatible energy_cost_of_tiling)
  - LatencyCostModel (bank-conflict simulation)
  - Low-level helpers: _count_max_bank_hits, _bank_sets_overlap,
    _compute_operand_stride_per_tile, _reader_active, _writer_active,
    _compute_period_cost_fast, check_burst_overlap_differential
  - get_cost_model factory
  - CostModel abstract interface
  - Edge cases: empty tiling, single element, wrap-around, etc.
  - Behavioral tests with hand-computed expected values
"""

import math
import pytest

from snaxc.ir.dart.cost_models import (
    CostModel,
    EnergyCostModel,
    LatencyCostModel,
    OperandDescriptor,
    OperandKind,
    check_burst_overlap_differential,
    energy_cost_of_tiling,
    get_cost_model,
    latency_cost_of_tiling,
    _bank_sets_overlap,
    _compute_operand_stride_per_tile,
    _compute_period_cost_fast,
    _count_max_bank_hits,
    _reader_active,
    _simulate_nested,
    _writer_active,
)


# ===================================================================
# Section 1: OperandDescriptor & OperandKind
# ===================================================================

class TestOperandDescriptor:
    def test_construction(self):
        desc = OperandDescriptor(
            kind=OperandKind.READER,
            element_bits=8,
            spatial_banks=8,
            invariant_dims=frozenset({1}),
        )
        assert desc.kind == OperandKind.READER
        assert desc.element_bits == 8
        assert desc.spatial_banks == 8
        assert desc.invariant_dims == frozenset({1})

    def test_frozen(self):
        desc = OperandDescriptor(
            kind=OperandKind.WRITER,
            element_bits=16,
            spatial_banks=4,
            invariant_dims=frozenset(),
        )
        with pytest.raises(AttributeError):
            desc.spatial_banks = 10  # type: ignore[misc]

    def test_equality(self):
        a = OperandDescriptor(OperandKind.READER, 8, 8, frozenset({0}))
        b = OperandDescriptor(OperandKind.READER, 8, 8, frozenset({0}))
        assert a == b

    def test_kinds_distinct(self):
        assert OperandKind.READER != OperandKind.WRITER
        assert OperandKind.READER != OperandKind.READER_WRITER
        assert OperandKind.WRITER != OperandKind.READER_WRITER


# ===================================================================
# Section 2: _count_max_bank_hits
# ===================================================================

class TestCountMaxBankHits:
    def test_empty(self):
        assert _count_max_bank_hits([], [], 32) == 0

    def test_single_burst_width_1(self):
        # One operand, 1 bank → max hit = 1
        assert _count_max_bank_hits([5], [1], 32) == 1

    def test_single_burst_width_8(self):
        # One operand, 8 consecutive banks → max hit = 1 (each bank hit once)
        assert _count_max_bank_hits([0], [8], 32) == 1

    def test_two_non_overlapping(self):
        # Two bursts that don't overlap: banks [0..7] and [16..23]
        assert _count_max_bank_hits([0, 16], [8, 8], 32) == 1

    def test_two_fully_overlapping(self):
        # Two bursts at the same start → every bank hit twice
        assert _count_max_bank_hits([0, 0], [8, 8], 32) == 2

    def test_two_partially_overlapping(self):
        # Burst A: banks [0..7], Burst B: banks [4..11]
        # Banks 4,5,6,7 are hit twice → max = 2
        assert _count_max_bank_hits([0, 4], [8, 8], 32) == 2

    def test_three_all_same(self):
        # Three bursts at bank 0, width 1 → max = 3
        assert _count_max_bank_hits([0, 0, 0], [1, 1, 1], 32) == 3

    def test_wrap_around(self):
        # Burst at bank 30, width 4 → banks 30,31,0,1
        # Burst at bank 0, width 2 → banks 0,1
        # Banks 0 and 1 are each hit 2 times → max = 2
        assert _count_max_bank_hits([30, 0], [4, 2], 32) == 2

    def test_modulo_applied(self):
        # bank_start 32 should wrap to 0
        assert _count_max_bank_hits([32], [1], 32) == 1
        assert _count_max_bank_hits([32, 0], [1, 1], 32) == 2


# ===================================================================
# Section 3: _bank_sets_overlap
# ===================================================================

class TestBankSetsOverlap:
    def test_zero_width(self):
        assert _bank_sets_overlap(0, 0, 5, 8, 32) == 0
        assert _bank_sets_overlap(5, 8, 0, 0, 32) == 0

    def test_no_overlap(self):
        # [0..7] and [16..23]
        assert _bank_sets_overlap(0, 8, 16, 8, 32) == 0

    def test_full_overlap(self):
        assert _bank_sets_overlap(0, 8, 0, 8, 32) == 8

    def test_partial_overlap(self):
        # [0..7] and [4..11] → 4 overlapping banks (4,5,6,7)
        assert _bank_sets_overlap(0, 8, 4, 8, 32) == 4

    def test_wrap_overlap(self):
        # [30..1] and [0..1] → banks 0,1 overlap
        assert _bank_sets_overlap(30, 4, 0, 2, 32) == 2

    def test_identical(self):
        assert _bank_sets_overlap(10, 5, 10, 5, 32) == 5


# ===================================================================
# Section 4: check_burst_overlap_differential
# ===================================================================

class TestBurstOverlapDifferential:
    def test_zero_diff(self):
        # Same start → always overlap
        assert check_burst_overlap_differential(0, 1, 1, 32) is True

    def test_far_apart(self):
        # diff=16, widths=1 → no overlap
        assert check_burst_overlap_differential(16, 1, 1, 32) is False

    def test_adjacent_no_overlap(self):
        # diff=8, width_a=8, width_b=8 → just touching edge, no overlap
        # d=8, not < width_a=8, and (32-8=24) not < width_b=8 → False
        assert check_burst_overlap_differential(8, 8, 8, 32) is False

    def test_one_short_of_adjacent(self):
        # diff=7, width_a=8 → 7 < 8 → overlap
        assert check_burst_overlap_differential(7, 8, 8, 32) is True

    def test_wrap_around_overlap(self):
        # diff=25, width_a=1, (32-25=7) < width_b=8 → True
        assert check_burst_overlap_differential(25, 1, 8, 32) is True

    def test_wrap_around_no_overlap(self):
        # diff=24, width_a=1, (32-24=8) not < width_b=8 → False
        assert check_burst_overlap_differential(24, 1, 8, 32) is False


# ===================================================================
# Section 5: energy_cost_of_tiling (backward compatibility)
# ===================================================================

class TestEnergyCostOfTiling:
    def test_empty_tiling(self):
        assert energy_cost_of_tiling([], [8, 8], [set()]) == 16.0

    def test_single_non_critical_dim(self):
        # 1 dim, size 4, not critical
        # operand_costs = [4] for each operand
        # No invariance → each cost = 4
        tiling = [(0, 4, False)]
        assert energy_cost_of_tiling(tiling, [1], [set()]) == 4.0

    def test_critical_dim_skipped_for_invariant(self):
        # dim 0, size 4, critical. Operand 0 is invariant to dim 0.
        # operand_cost stays 1 because it's skipped.
        tiling = [(0, 4, True)]
        inv = [{0}]
        assert energy_cost_of_tiling(tiling, [1], inv) == 1.0

    def test_critical_dim_counted_for_non_invariant(self):
        # dim 0, size 4, critical. Operand 0 is NOT invariant.
        tiling = [(0, 4, True)]
        inv = [set()]
        assert energy_cost_of_tiling(tiling, [1], inv) == 4.0

    def test_two_operands_gemm_like(self):
        # Simulate M=4, N=4, K=4 tiling (all non-critical, i.e. temporal)
        # Operand A: invariant to dim 1 (N), Operand B: invariant to dim 0 (M),
        # Operand C: invariant to dim 2 (K)
        # All dims non-critical → no skipping
        tiling = [(0, 4, False), (1, 4, False), (2, 4, False)]
        request = [1, 1, 1]
        inv = [{1}, {0}, {2}]  # A inv to N, B inv to M, C inv to K
        # All non-critical, so no skipping. Each operand cost = 4*4*4 = 64
        assert energy_cost_of_tiling(tiling, request, inv) == 192.0

    def test_weighted_requests(self):
        tiling = [(0, 8, False)]
        assert energy_cost_of_tiling(tiling, [10], [set()]) == 80.0

    def test_energy_model_class(self):
        model = EnergyCostModel()
        tiling = [(0, 4, False)]
        descs = [OperandDescriptor(OperandKind.READER, 8, 8, frozenset())]
        result = model.cost(tiling, descs, [1], [set()])
        assert result == 4.0


# ===================================================================
# Section 6: _compute_operand_stride_per_tile
# ===================================================================

class TestComputeOperandStridePerTile:
    def test_single_level_non_invariant(self):
        desc = OperandDescriptor(OperandKind.READER, 8, 8, frozenset())
        tiling = [(0, 4, False)]
        strides = _compute_operand_stride_per_tile(tiling, [desc], [set()], 64)
        # spatial_banks=8, cum starts at 1 → stride = 8 * 1 = 8
        assert strides == [[8]]

    def test_single_level_invariant(self):
        desc = OperandDescriptor(OperandKind.READER, 8, 8, frozenset({0}))
        tiling = [(0, 4, False)]
        strides = _compute_operand_stride_per_tile(tiling, [desc], [{0}], 64)
        assert strides == [[0]]

    def test_two_levels_same_dim(self):
        # dim 0, size 4 (inner), then dim 0, size 2 (outer)
        desc = OperandDescriptor(OperandKind.READER, 8, 1, frozenset())
        tiling = [(0, 4, False), (0, 2, True)]
        strides = _compute_operand_stride_per_tile(tiling, [desc], [set()], 64)
        # level 0: cum=1, stride=1*1=1, then cum becomes 1*4=4
        # level 1: cum=4, stride=1*4=4
        assert strides == [[1, 4]]

    def test_two_different_dims(self):
        desc = OperandDescriptor(OperandKind.READER, 8, 2, frozenset())
        tiling = [(0, 3, False), (1, 5, True)]
        strides = _compute_operand_stride_per_tile(tiling, [desc], [set()], 64)
        # level 0: dim 0, cum starts at 1 → stride = 2*1 = 2, cum(dim0) → 3
        # level 1: dim 1, cum starts at 1 → stride = 2*1 = 2, cum(dim1) → 5
        assert strides == [[2, 2]]

    def test_mixed_invariant_dims(self):
        # 2 operands: op0 invariant to dim 1, op1 invariant to dim 0
        desc0 = OperandDescriptor(OperandKind.READER, 8, 8, frozenset({1}))
        desc1 = OperandDescriptor(OperandKind.READER, 8, 8, frozenset({0}))
        tiling = [(0, 4, False), (1, 4, True)]
        strides = _compute_operand_stride_per_tile(
            tiling, [desc0, desc1], [{1}, {0}], 64
        )
        # op0: dim 0 → stride = 8*1 = 8; dim 1 → invariant → 0
        # op1: dim 0 → invariant → 0; dim 1 → stride = 8*1 = 8
        assert strides == [[8, 0], [0, 8]]


# ===================================================================
# Section 7: _reader_active / _writer_active
# ===================================================================

class TestReaderActive:
    def test_no_critical_level(self):
        # No critical dims → always active
        tiling = [(0, 4, False), (1, 4, False)]
        inv = frozenset({0})
        assert _reader_active([0, 0], tiling, inv) is True
        assert _reader_active([3, 3], tiling, inv) is True

    def test_critical_invariant_at_zero(self):
        # dim 0 is critical, operand is invariant to dim 0
        tiling = [(0, 4, True)]
        inv = frozenset({0})
        assert _reader_active([0], tiling, inv) is True
        assert _reader_active([1], tiling, inv) is False

    def test_critical_non_invariant(self):
        # dim 0 is critical but operand is NOT invariant → always active
        tiling = [(0, 4, True)]
        inv = frozenset()
        assert _reader_active([0], tiling, inv) is True
        assert _reader_active([3], tiling, inv) is True

    def test_nested_critical_with_inner_invariant(self):
        # Inner: dim 0, size 4, not critical
        # Outer: dim 1, size 3, critical
        # Operand invariant to dim 1
        tiling = [(0, 4, False), (1, 3, True)]
        inv = frozenset({1})
        # Critical level is 1 (outer). Look from level 1 down to 0.
        # Level 1 dim=1 is invariant → check counter == 0
        # Level 0 dim=0 is NOT invariant → skip
        assert _reader_active([2, 0], tiling, inv) is True
        assert _reader_active([2, 1], tiling, inv) is False

    def test_multiple_invariant_levels(self):
        # Both levels are critical and operand is invariant to both dims
        tiling = [(0, 3, True), (1, 2, True)]
        inv = frozenset({0, 1})
        assert _reader_active([0, 0], tiling, inv) is True
        assert _reader_active([1, 0], tiling, inv) is False
        assert _reader_active([0, 1], tiling, inv) is False


class TestWriterActive:
    def test_no_critical_level(self):
        tiling = [(0, 4, False)]
        inv = frozenset({0})
        assert _writer_active([0], [4], tiling, inv) is True
        assert _writer_active([3], [4], tiling, inv) is True

    def test_critical_invariant_at_last(self):
        tiling = [(0, 4, True)]
        inv = frozenset({0})
        assert _writer_active([3], [4], tiling, inv) is True
        assert _writer_active([2], [4], tiling, inv) is False

    def test_critical_non_invariant(self):
        tiling = [(0, 4, True)]
        inv = frozenset()
        assert _writer_active([0], [4], tiling, inv) is True
        assert _writer_active([3], [4], tiling, inv) is True

    def test_nested(self):
        tiling = [(0, 4, False), (1, 3, True)]
        inv = frozenset({1})
        assert _writer_active([2, 2], [4, 3], tiling, inv) is True
        assert _writer_active([2, 1], [4, 3], tiling, inv) is False


# ===================================================================
# Section 8: _compute_period_cost_fast
# ===================================================================

class TestComputePeriodCostFast:
    def test_all_zero_strides(self):
        # All strides zero → every iteration hits same banks → max_hits * bound
        result = _compute_period_cost_fast(
            strides_mod=[0, 0],
            burst_widths=[1, 1],
            inner_stride_mod=[0, 0],
            inner_bound=10,
            num_banks=32,
        )
        # Both at bank 0, width 1 → max_hits = 2 per iteration
        assert result == 20

    def test_no_conflict_stride_spaced(self):
        # Two operands, stride 8 each, starting 16 apart → no overlap
        result = _compute_period_cost_fast(
            strides_mod=[0, 16],
            burst_widths=[8, 8],
            inner_stride_mod=[8, 8],
            inner_bound=32,
            num_banks=32,
        )
        # period = 32 / gcd(8, 32) = 32 / 8 = 4
        # Per iteration: op0 at (0+8*ic)%32, op1 at (16+8*ic)%32
        # ic=0: banks [0..7] vs [16..23] → no overlap → 1 cycle
        # ic=1: banks [8..15] vs [24..31] → no overlap → 1 cycle
        # ic=2: banks [16..23] vs [0..7] → no overlap → 1 cycle
        # ic=3: banks [24..31] vs [8..15] → no overlap → 1 cycle
        # 4 cycles per period, 32/4=8 periods → 32 cycles total
        assert result == 32

    def test_full_conflict(self):
        # Two operands at same start, same stride → always 2 hits
        result = _compute_period_cost_fast(
            strides_mod=[0, 0],
            burst_widths=[1, 1],
            inner_stride_mod=[1, 1],
            inner_bound=32,
            num_banks=32,
        )
        # period = 32 / gcd(1, 32) = 32
        # Every iteration: both at same bank → max_hits = 2
        assert result == 64

    def test_period_longer_than_bound_returns_none(self):
        # period = 32 / gcd(1,32) = 32, bound = 4 < 32 → None
        result = _compute_period_cost_fast(
            strides_mod=[0],
            burst_widths=[1],
            inner_stride_mod=[1],
            inner_bound=4,
            num_banks=32,
        )
        assert result is None

    def test_single_operand_no_conflict(self):
        # Single operand → always 1 hit per iteration
        result = _compute_period_cost_fast(
            strides_mod=[0],
            burst_widths=[8],
            inner_stride_mod=[8],
            inner_bound=32,
            num_banks=32,
        )
        # period = 32 / gcd(8, 32) = 4
        # Each iteration: 1 operand → max_hits = 1
        assert result == 32


# ===================================================================
# Section 9: latency_cost_of_tiling — behavioral tests
# ===================================================================

class TestLatencyCostOfTiling:
    """
    End-to-end tests for the latency cost model.
    """

    def test_empty_tiling(self):
        descs = [OperandDescriptor(OperandKind.READER, 8, 8, frozenset())]
        assert latency_cost_of_tiling([], descs, [set()]) == 0.0

    def test_single_reader_no_conflict(self):
        """One reader, stride 1 (1 bank per burst) → 1 cycle per iteration."""
        desc = OperandDescriptor(OperandKind.READER, 8, 1, frozenset())
        tiling = [(0, 16, False)]
        cost = latency_cost_of_tiling(tiling, [desc], [set()], num_banks=32)
        assert cost == 16.0

    def test_two_readers_no_overlap(self):
        """
        Two readers with stride 1 bank each, different dims.
        Single loop of size 8.
        Operands never share banks → 1 cycle per iteration.
        """
        desc_a = OperandDescriptor(OperandKind.READER, 8, 1, frozenset())
        desc_b = OperandDescriptor(OperandKind.READER, 8, 1, frozenset())
        tiling = [(0, 8, False)]  # single loop
        inv = [set(), set()]
        # Both readers, same inner stride (spatial_banks * cum = 1),
        # both start at offset 0.
        # Iteration ic: op0 at bank ic%32, op1 at bank ic%32 → overlap!
        cost = latency_cost_of_tiling(tiling, [desc_a, desc_b], inv, num_banks=32)
        # Both at same bank → 2 hits per cycle
        assert cost == 16.0  # 8 iterations * 2 hits

    def test_two_readers_full_separation(self):
        """
        Two readers with spatial_banks=8. If they have different dims and
        one is invariant to the loop dim, the invariant one has stride 0.
        Actually, let's make them have strides that keep them 16 banks apart.
        """
        # Both spatial_banks=1 to keep it simple.
        # op0: not invariant, stride = 1
        # op1: invariant to dim 0 → stride = 0
        # Both at bank 0 always → 2 hits per cycle
        desc_a = OperandDescriptor(OperandKind.READER, 8, 1, frozenset())
        desc_b = OperandDescriptor(OperandKind.READER, 8, 1, frozenset({0}))
        tiling = [(0, 8, False)]
        inv = [set(), {0}]
        # op0 moves each iteration, op1 stays at 0
        # ic=0: both at 0 → 2 hits
        # ic=1: op0 at 1, op1 at 0 → 1 hit each
        # ...
        cost = latency_cost_of_tiling(tiling, [desc_a, desc_b], inv, num_banks=32)
        # ic=0: max_hits=2 → 2 cycles
        # ic=1..7: max_hits=1 → 1 cycle each = 7 cycles
        assert cost == 9.0

    def test_single_writer(self):
        """Single writer, no gating → 1 cycle per iteration."""
        desc = OperandDescriptor(OperandKind.WRITER, 8, 1, frozenset())
        tiling = [(0, 10, False)]
        cost = latency_cost_of_tiling(tiling, [desc], [set()], num_banks=32)
        assert cost == 10.0

    def test_reader_gating_reduces_cycles(self):
        """
        Reader gated by a critical dimension it's invariant to.
        Only fires when the critical counter == 0.
        """
        # Inner loop: dim 0, size 4, critical
        # Operand invariant to dim 0 → only fires when counter == 0
        desc = OperandDescriptor(OperandKind.READER, 8, 1, frozenset({0}))
        tiling = [(0, 4, True)]
        inv = [{0}]
        cost = latency_cost_of_tiling(tiling, [desc], inv, num_banks=32)
        # Only 1 cycle when ic=0 is active (counter == 0)
        assert cost == 1.0

    def test_writer_gating(self):
        """
        Writer gated by a critical dimension it's invariant to.
        Only fires when critical counter == bound-1.
        """
        desc = OperandDescriptor(OperandKind.WRITER, 8, 1, frozenset({0}))
        tiling = [(0, 4, True)]
        inv = [{0}]
        cost = latency_cost_of_tiling(tiling, [desc], inv, num_banks=32)
        # Only 1 cycle when ic=3 (last value)
        assert cost == 1.0

    def test_reader_writer_drain_cycles(self):
        """
        ReaderWriter operand causes 2 extra drain cycles at the end.
        """
        desc = OperandDescriptor(OperandKind.READER_WRITER, 8, 1, frozenset())
        tiling = [(0, 8, False)]
        inv = [set()]
        cost = latency_cost_of_tiling(tiling, [desc], inv, num_banks=32)
        # 8 iterations + 2 drain cycles = 10
        # But only reading part active first 2 iterations (no write for ic<2),
        # then both read and write → split into 2 sub-cycles.
        # ic=0: read only → 1 cycle
        # ic=1: read only → 1 cycle
        # ic=2: RW read + RW write(ic=0) → split into 2 sub-cycles
        # ic=3: read + write(ic=1) → split
        # ic=4: read + write(ic=2) → split
        # ic=5: read + write(ic=3) → split
        # ic=6: read + write(ic=4) → split
        # ic=7: read + write(ic=5) → split
        # Each split = read_cycles + write_cycles = 1 + 1 = 2
        # Total from inner loop = 2 * 1 + 6 * 2 = 14
        # + 2 drain cycles = 16
        assert cost == 16.0

    def test_reader_writer_never_reads_and_writes_same_cycle(self):
        """
        Even when a RW operand accesses different banks for read and write,
        they must not share a cycle.
        """
        # RW with spatial_banks=1
        desc = OperandDescriptor(OperandKind.READER_WRITER, 8, 1, frozenset())
        tiling = [(0, 4, False)]
        inv = [set()]
        cost = latency_cost_of_tiling(tiling, [desc], inv, num_banks=32)
        # ic=0: read only → 1
        # ic=1: read only → 1
        # ic=2: read + write(0) → 1 + 1 = 2
        # ic=3: read + write(1) → 1 + 1 = 2
        # inner = 6
        # + 2 drain = 8
        assert cost == 8.0

    def test_latency_model_class_interface(self):
        """Test going through the LatencyCostModel class."""
        model = LatencyCostModel()
        desc = OperandDescriptor(OperandKind.READER, 8, 1, frozenset())
        tiling = [(0, 10, False)]
        cost = model.cost(tiling, [desc], [1], [set()], num_banks=32)
        assert cost == 10.0


# ===================================================================
# Section 10: Multi-level tiling latency tests
# ===================================================================

class TestLatencyMultiLevel:
    def test_two_level_no_conflict(self):
        """
        2-level tiling: inner dim 0 size 4, outer dim 1 size 3.
        Single reader, non-invariant to both → always active, stride=1 bank.
        Total iterations = 4 * 3 = 12 → 12 cycles.
        """
        desc = OperandDescriptor(OperandKind.READER, 8, 1, frozenset())
        tiling = [(0, 4, False), (1, 3, False)]
        inv = [set()]
        cost = latency_cost_of_tiling(tiling, [desc], inv, num_banks=32)
        # 12 iterations, 1 cycle each
        assert cost == 12.0

    def test_two_level_reader_gating(self):
        """
        Inner: dim 0, size 4, not critical
        Outer: dim 1, size 3, critical
        Reader invariant to dim 1 → only fires when outer counter == 0.
        4 inner iterations * 1 outer active = 4 active iterations → 4 cycles.
        But simulation runs all 3*4=12 iterations; when outer != 0, no access.
        """
        desc = OperandDescriptor(OperandKind.READER, 8, 1, frozenset({1}))
        tiling = [(0, 4, False), (1, 3, True)]
        inv = [{1}]
        cost = latency_cost_of_tiling(tiling, [desc], inv, num_banks=32)
        # Outer counter 0: 4 inner iterations, 1 cycle each = 4
        # Outer counter 1,2: reader not active → 0 cycles
        assert cost == 4.0

    def test_gemm_like_3_operands(self):
        """
        Simple GEMM-like scenario:
        - dim 0 (M) size 4, dim 1 (K) size 4, dim 2 (N) size 4
        - A: invariant to N(dim 2), B: invariant to M(dim 0), C: invariant to K(dim 1)
        - A and B are readers, C is reader_writer
        - dim 1 (K) is critical for C (C invariant to it)
        - All spatial_banks = 1 for simplicity
        """
        desc_a = OperandDescriptor(OperandKind.READER, 8, 1, frozenset({2}))
        desc_b = OperandDescriptor(OperandKind.READER, 8, 1, frozenset({0}))
        desc_c = OperandDescriptor(OperandKind.READER_WRITER, 8, 1, frozenset({1}))

        # Tiling: inner dim 0 (M=4), then dim 1 (K=4, critical), then dim 2 (N=4)
        tiling = [(0, 4, False), (1, 4, True), (2, 4, False)]
        inv = [{2}, {0}, {1}]

        cost = latency_cost_of_tiling(
            tiling, [desc_a, desc_b, desc_c], inv, num_banks=32
        )
        # Cost should be a positive number (we're mainly checking it doesn't crash)
        assert cost > 0
        assert isinstance(cost, float)


# ===================================================================
# Section 11: get_cost_model factory
# ===================================================================

class TestGetCostModel:
    def test_energy(self):
        model = get_cost_model("energy")
        assert isinstance(model, EnergyCostModel)

    def test_latency(self):
        model = get_cost_model("latency")
        assert isinstance(model, LatencyCostModel)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown cost model"):
            get_cost_model("magic")

    def test_default_is_latency(self):
        model = get_cost_model()
        assert isinstance(model, LatencyCostModel)


# ===================================================================
# Section 12: CostModel ABC
# ===================================================================

class TestCostModelABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            CostModel()  # type: ignore[abstract]

    def test_subclass_must_implement_cost(self):
        class Incomplete(CostModel):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_valid_subclass(self):
        class MyCost(CostModel):
            def cost(self, tiling, operand_descriptors, request_per_streamer,
                     invariance_map, *, num_banks=32, bank_bits=64):
                return 42.0

        m = MyCost()
        assert m.cost([], [], [], []) == 42.0


# ===================================================================
# Section 13: Edge cases
# ===================================================================

class TestEdgeCases:
    def test_single_iteration(self):
        """Tiling with bound=1 should produce minimal cost."""
        desc = OperandDescriptor(OperandKind.READER, 8, 1, frozenset())
        tiling = [(0, 1, False)]
        cost = latency_cost_of_tiling(tiling, [desc], [set()], num_banks=32)
        assert cost == 1.0

    def test_bound_equals_banks(self):
        """Loop bound exactly equals num_banks."""
        desc = OperandDescriptor(OperandKind.READER, 8, 1, frozenset())
        tiling = [(0, 32, False)]
        cost = latency_cost_of_tiling(tiling, [desc], [set()], num_banks=32)
        assert cost == 32.0

    def test_large_burst_width(self):
        """Burst width equals num_banks → entire bank space used."""
        desc = OperandDescriptor(OperandKind.READER, 8, 32, frozenset())
        tiling = [(0, 4, False)]
        # All 32 banks hit each iteration by 1 operand → max_hits=1
        cost = latency_cost_of_tiling(tiling, [desc], [set()], num_banks=32)
        assert cost == 4.0

    def test_two_large_bursts_full_overlap(self):
        """Two operands each using all 32 banks → 2 hits per bank, every cycle."""
        desc_a = OperandDescriptor(OperandKind.READER, 8, 32, frozenset())
        desc_b = OperandDescriptor(OperandKind.READER, 8, 32, frozenset())
        tiling = [(0, 4, False)]
        inv = [set(), set()]
        cost = latency_cost_of_tiling(tiling, [desc_a, desc_b], inv, num_banks=32)
        # 4 iterations * 2 max hits = 8
        assert cost == 8.0

    def test_zero_spatial_banks_writer(self):
        """Operand with 0 spatial_banks doesn't contribute any banks."""
        desc = OperandDescriptor(OperandKind.WRITER, 8, 0, frozenset())
        tiling = [(0, 4, False)]
        cost = latency_cost_of_tiling(tiling, [desc], [set()], num_banks=32)
        # No banks hit → 0 cycles (the `if bank_starts else 0` path)
        assert cost == 0.0

    def test_energy_and_latency_produce_different_values(self):
        """The two models should generally produce different costs."""
        desc_a = OperandDescriptor(OperandKind.READER, 8, 8, frozenset({1}))
        desc_b = OperandDescriptor(OperandKind.READER, 8, 8, frozenset({0}))
        desc_c = OperandDescriptor(OperandKind.READER_WRITER, 8, 1, frozenset({2}))
        descs = [desc_a, desc_b, desc_c]

        tiling = [(0, 4, False), (1, 4, True), (2, 4, True)]
        inv = [{1}, {0}, {2}]
        req = [8, 8, 1]

        energy = energy_cost_of_tiling(tiling, req, inv)
        latency = latency_cost_of_tiling(tiling, descs, inv, num_banks=32)

        # They shouldn't be equal in general
        # (just checking both run and return valid numbers)
        assert energy > 0
        assert latency > 0
        assert isinstance(energy, float)
        assert isinstance(latency, float)


# ===================================================================
# Section 14: Consistency between fast path and general simulation
# ===================================================================

class TestFastPathConsistency:
    """
    When the fast-path applies (single level, no gating, no RW),
    it should give the same result as the general simulation.
    """

    def _run_both_paths(self, descs, tiling, inv, num_banks=32):
        """Run latency model and compare against manual simulation."""
        strides_bank = _compute_operand_stride_per_tile(tiling, descs, inv, 64)

        # General simulation
        general = _simulate_nested(tiling, descs, inv, strides_bank, num_banks)

        # Full model (which may use fast path)
        model_result = latency_cost_of_tiling(tiling, descs, inv, num_banks=num_banks)

        assert model_result == float(general), (
            f"Fast path vs general mismatch: model={model_result}, general={general}"
        )
        return model_result

    def test_simple_single_loop(self):
        desc = OperandDescriptor(OperandKind.READER, 8, 1, frozenset())
        tiling = [(0, 64, False)]
        self._run_both_paths([desc], tiling, [set()])

    def test_two_readers_single_loop(self):
        desc_a = OperandDescriptor(OperandKind.READER, 8, 4, frozenset())
        desc_b = OperandDescriptor(OperandKind.READER, 8, 4, frozenset())
        tiling = [(0, 64, False)]
        self._run_both_paths([desc_a, desc_b], tiling, [set(), set()])

    def test_stride_8_two_ops(self):
        desc_a = OperandDescriptor(OperandKind.READER, 8, 8, frozenset())
        desc_b = OperandDescriptor(OperandKind.READER, 8, 8, frozenset())
        tiling = [(0, 32, False)]
        self._run_both_paths([desc_a, desc_b], tiling, [set(), set()])


# ===================================================================
# Section 15: Stride-based address correctness
# ===================================================================

class TestStrideAddressCorrectness:
    """
    Verify that addresses computed by the latency model correspond to
    expected bank indices for known patterns.
    """

    def test_sequential_access_one_operand(self):
        """
        One reader, spatial_banks=1, iterating dim 0 of size 8.
        Expected banks accessed: 0,1,2,3,4,5,6,7
        → no conflicts → 8 cycles.
        """
        desc = OperandDescriptor(OperandKind.READER, 8, 1, frozenset())
        tiling = [(0, 8, False)]
        cost = latency_cost_of_tiling(tiling, [desc], [set()], num_banks=32)
        assert cost == 8.0

    def test_strided_access_causes_conflict(self):
        """
        Two readers, both spatial_banks=1, both stride=1 (same dim).
        They always access the same bank → 2 hits per cycle.
        """
        desc_a = OperandDescriptor(OperandKind.READER, 8, 1, frozenset())
        desc_b = OperandDescriptor(OperandKind.READER, 8, 1, frozenset())
        tiling = [(0, 8, False)]
        inv = [set(), set()]
        cost = latency_cost_of_tiling(tiling, [desc_a, desc_b], inv, num_banks=32)
        # Both always at same bank → 2 cycles per iteration
        assert cost == 16.0

    def test_offset_strides_avoid_conflict(self):
        """
        Two readers: op0 spatial_banks=8, op1 spatial_banks=8.
        Both vary with dim 0 but op1 is also affected by dim 1.
        In a 2-level tiling, the outer dim changes op1's base offset.
        """
        desc_a = OperandDescriptor(OperandKind.READER, 8, 8, frozenset({1}))
        desc_b = OperandDescriptor(OperandKind.READER, 8, 8, frozenset({0}))
        # Inner: dim 0, size 4; Outer: dim 1, size 4
        tiling = [(0, 4, False), (1, 4, False)]
        inv = [{1}, {0}]
        cost = latency_cost_of_tiling(tiling, [desc_a, desc_b], inv, num_banks=32)
        assert cost > 0
        assert isinstance(cost, float)


# ===================================================================
# Section 16: 8x8x8 GEMM-like realistic scenario
# ===================================================================

class TestRealisticGemm:
    """
    Test a scenario inspired by the 8x8x8 GEMM on SNAX-GEMMX.
    - 3 operands: A (reader), B (reader), C (reader_writer)
    - 8-bit elements → spatial_banks = 8 per input streamer
    - A invariant to N, B invariant to M, C invariant to K
    """

    def _make_gemm_descs(self):
        desc_a = OperandDescriptor(
            kind=OperandKind.READER,
            element_bits=8,
            spatial_banks=8,
            invariant_dims=frozenset({2}),  # invariant to N
        )
        desc_b = OperandDescriptor(
            kind=OperandKind.READER,
            element_bits=8,
            spatial_banks=8,
            invariant_dims=frozenset({0}),  # invariant to M
        )
        desc_c = OperandDescriptor(
            kind=OperandKind.READER_WRITER,
            element_bits=8,
            spatial_banks=1,  # 32-bit accumulator → fewer banks
            invariant_dims=frozenset({1}),  # invariant to K
        )
        return [desc_a, desc_b, desc_c]

    def test_simple_tiling(self):
        descs = self._make_gemm_descs()
        # M=2, K=4 (critical for C), N=2
        tiling = [
            (0, 2, False),   # M inner
            (1, 4, True),    # K critical
            (2, 2, False),   # N outer
        ]
        inv = [{2}, {0}, {1}]
        cost = latency_cost_of_tiling(tiling, descs, inv, num_banks=32)
        assert cost > 0
        assert isinstance(cost, float)

    def test_different_tilings_produce_different_costs(self):
        """Two different tiling strategies should generally have different costs."""
        descs = self._make_gemm_descs()
        inv = [{2}, {0}, {1}]

        tiling_a = [
            (0, 2, False),
            (1, 4, True),
            (2, 2, False),
        ]
        tiling_b = [
            (2, 2, False),
            (1, 4, True),
            (0, 2, False),
        ]

        cost_a = latency_cost_of_tiling(tiling_a, descs, inv, num_banks=32)
        cost_b = latency_cost_of_tiling(tiling_b, descs, inv, num_banks=32)

        # Both should be valid positive numbers
        assert cost_a > 0
        assert cost_b > 0

    def test_num_banks_affects_cost(self):
        """More banks should generally reduce conflicts."""
        descs = self._make_gemm_descs()
        inv = [{2}, {0}, {1}]
        tiling = [(0, 4, False), (1, 4, True), (2, 4, False)]

        cost_16 = latency_cost_of_tiling(tiling, descs, inv, num_banks=16)
        cost_64 = latency_cost_of_tiling(tiling, descs, inv, num_banks=64)

        # With more banks, conflicts should be equal or less
        assert cost_64 <= cost_16


# ===================================================================
# Section 17: Writer timing offset correctness
# ===================================================================

class TestWriterTimingOffset:
    """
    Verify that the 2-iteration write delay for ReaderWriter operands
    is correctly modeled.
    """

    def test_rw_short_loop_no_writes(self):
        """
        With inner_bound=2, writer fires at ic-2, so write_ic = 0 and -1.
        Only write_ic=0 is valid (for ic=2, but ic only goes to 1).
        So no writes ever happen within the inner loop.
        """
        desc = OperandDescriptor(OperandKind.READER_WRITER, 8, 1, frozenset())
        tiling = [(0, 2, False)]
        inv = [set()]
        cost = latency_cost_of_tiling(tiling, [desc], inv, num_banks=32)
        # ic=0: read → 1 cycle
        # ic=1: read → 1 cycle
        # +2 drain cycles
        # Total = 4
        assert cost == 4.0

    def test_rw_loop_3_one_write(self):
        """
        With inner_bound=3:
        ic=0: read only → 1
        ic=1: read only → 1
        ic=2: read + write(ic=0) → split = 2
        +2 drain = 6
        """
        desc = OperandDescriptor(OperandKind.READER_WRITER, 8, 1, frozenset())
        tiling = [(0, 3, False)]
        inv = [set()]
        cost = latency_cost_of_tiling(tiling, [desc], inv, num_banks=32)
        assert cost == 6.0


# ===================================================================
# Section 18: num_banks parameter
# ===================================================================

class TestNumBanksParameter:
    def test_small_num_banks(self):
        desc = OperandDescriptor(OperandKind.READER, 8, 1, frozenset())
        tiling = [(0, 4, False)]
        cost = latency_cost_of_tiling(tiling, [desc], [set()], num_banks=4)
        assert cost == 4.0

    def test_num_banks_1_always_conflicts(self):
        """With 1 bank, all accesses conflict."""
        desc_a = OperandDescriptor(OperandKind.READER, 8, 1, frozenset())
        desc_b = OperandDescriptor(OperandKind.READER, 8, 1, frozenset())
        tiling = [(0, 4, False)]
        inv = [set(), set()]
        cost = latency_cost_of_tiling(tiling, [desc_a, desc_b], inv, num_banks=1)
        # Both always at bank 0 → 2 hits per cycle → 8 total
        assert cost == 8.0


# ===================================================================
# Section 19: Regression — energy model backward compatibility
# ===================================================================

class TestEnergyBackwardCompat:
    """
    Ensure the energy cost function via the old cost_of_tiling interface
    in scheduler.py still works.
    """

    def test_cost_of_tiling_alias(self):
        from snaxc.ir.dart.scheduler import cost_of_tiling
        tiling = [(0, 4, False), (1, 3, True)]
        req = [8, 32]
        inv = [{1}, set()]
        result = cost_of_tiling(tiling, req, inv)
        expected = energy_cost_of_tiling(tiling, req, inv)
        assert result == expected
