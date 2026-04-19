"""
ZigZag-based cost model adapter for the SNAX DART scheduler.

Translates snaxc's tiling representation into ZigZag's mapping format
and evaluates it using ZigZag's built-in CostModelEvaluation.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache
from math import prod
from typing import Any

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import Constants, LayerDim, LayerOperand, MemoryOperand, OADimension
from zigzag.hardware.architecture.accelerator import Accelerator
from zigzag.mapping.mapping_assist_funcs import SpatialMappingPerMemLvl
from zigzag.mapping.spatial_mapping import MappingSingleOADim, SpatialMapping, SpatialMappingHint
from zigzag.mapping.spatial_mapping_internal import SpatialMappingInternal
from zigzag.mapping.temporal_mapping import TemporalMapping, TemporalMappingDict, TemporalMappingType
from zigzag.parser.accelerator_factory import AcceleratorFactory
from zigzag.parser.accelerator_validator import AcceleratorValidator
from zigzag.stages.mapping.spatial_mapping_conversion import SpatialMappingConversionStage
from zigzag.utils import open_yaml
from zigzag.workload.layer_attributes import (
    InputOperandSource,
    LayerDimSizes,
    LayerEquation,
    LayerOperandPrecision,
    LayerPadding,
    LayerTemporalOrdering,
    MemoryOperandLinks,
)
from zigzag.workload.layer_node import LayerNode, LayerNodeAttributes, MappingAttributes

from snaxc.ir.dart.cost_models import OperandDescriptor

# ---------------------------------------------------------------------------
# Default hardware description path
# ---------------------------------------------------------------------------
_DEFAULT_HW_YAML = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",  # snaxc -> snax-mlir
    "..",             # snax-mlir -> workspace root
    "zigzag", "zigzag", "inputs", "hardware", "gemm_l1_l3_fixed_cache.yaml",
)

# ---------------------------------------------------------------------------
# Cached accelerator loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_accelerator(yaml_path: str) -> Accelerator:
    """Load and cache a ZigZag Accelerator from a YAML hardware description.

    The hardware YAML specifies all values in ZigZag's native units:
    sizes in **bits** and port bandwidths in **bits/cycle**.
    """
    data = open_yaml(yaml_path)
    validator = AcceleratorValidator(data)
    validated = validator.normalized_data
    if not validator.validate():
        raise ValueError(f"ZigZag HW validation failed: {validator.validator.errors}")
    factory = AcceleratorFactory(validated)
    return factory.create()


# ---------------------------------------------------------------------------
# Dimension and operand mapping helpers
# ---------------------------------------------------------------------------

# Template dim index → ZigZag LayerDim name.
# In the GEMMX accelerator the OA dimensions are D1, D2, D3 with sizes 8,8,8
# and the GEMM equation is O[D1][D2] += I[D1][D3] * W[D3][D2].
DIM_NAMES = {0: "D1", 1: "D2", 2: "D3"}

# ZigZag layer equation for a standard GEMM.
GEMM_EQUATION = "O[D1][D2] = I[D1][D3] * W[D3][D2]"

# Layer operands
_I = Constants.LAYER_OP_I   # Input  (A / I1)
_W = Constants.LAYER_OP_W   # Weight (B / I2)
_O = Constants.OUTPUT_LAYER_OP  # Output

# Memory operands
_MEM_I1 = Constants.MEM_OP_1
_MEM_I2 = Constants.MEM_OP_2
_MEM_O  = Constants.OUTPUT_MEM_OP

# Memory-operand → Layer-operand mapping
_MEM_TO_LAYER = {_MEM_I1: _I, _MEM_I2: _W, _MEM_O: _O}
_LAYER_TO_MEM = {_I: _MEM_I1, _W: _MEM_I2, _O: _MEM_O}


def _dim(idx: int) -> LayerDim:
    return LayerDim(DIM_NAMES[idx])


# ---------------------------------------------------------------------------
# Tiling → ZigZag temporal-mapping conversion
# ---------------------------------------------------------------------------

def _split_tiling_into_levels(
    tiling: list[tuple[int, int, bool]],
    invariance_map: list[set[int]],
    num_temporal_levels: dict[LayerOperand, int],
) -> TemporalMappingDict:
    """Convert an snaxc tiling (inner→outer) into a ZigZag TemporalMappingDict.

    The tiling is a flat list of (template_dim_idx, tile_size, is_critical).
    Critical flags mark cache-level boundaries.

    ``num_temporal_levels`` gives the number of temporal mapping levels per
    operand.  This equals the number of hardware memory levels for that
    operand (arch_level - 1 in ZigZag terms, since the lowest arch level
    is the MAC level which only holds spatial mapping).
    """
    all_ops = [_I, _W, _O]

    # Build per-operand invariance lookup
    inv_sets: dict[LayerOperand, set[int]] = {}
    op_order = [_I, _W, _O]
    for i, op in enumerate(op_order):
        if i < len(invariance_map):
            inv_sets[op] = invariance_map[i]
        else:
            inv_sets[op] = set()

    # Current memory level index per operand (start at innermost = 0)
    cur_level: dict[LayerOperand, int] = {op: 0 for op in all_ops}

    # Build the result: per operand, per temporal level
    result: dict[LayerOperand, list[list[tuple[LayerDim, int]]]] = {
        op: [[] for _ in range(num_temporal_levels[op])] for op in all_ops
    }

    for dim_idx, tile_size, is_critical in tiling:
        if tile_size <= 1:
            continue

        layer_dim = _dim(dim_idx)

        # Append this loop to the current level for each operand
        for op in all_ops:
            lvl = min(cur_level[op], num_temporal_levels[op] - 1)
            result[op][lvl].append((layer_dim, tile_size))

        # If this is a critical loop, bump the memory level for operands
        # that are invariant to this dimension.
        if is_critical:
            for op in all_ops:
                if dim_idx in inv_sets[op]:
                    cur_level[op] = min(cur_level[op] + 1, num_temporal_levels[op] - 1)

    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Spatial mapping construction
# ---------------------------------------------------------------------------

def _build_spatial_mapping_per_mem_lvl(
    accelerator: Accelerator,
    user_spatial_mapping: SpatialMapping,
    memory_operand_links: MemoryOperandLinks,
) -> SpatialMappingPerMemLvl:
    """Build ZigZag's internal spatial mapping dict from the accelerator's
    memory hierarchy and user-defined spatial mapping.

    Replicates the logic from ``SpatialMappingConversionStage.generate_mapping_per_mem_lvl``.
    Each OA dimension's unrolling is consumed by the first memory level
    that serves it, and any remaining unrolling spills to a top-level entry.
    """
    mapping_per_mem_lvl: SpatialMappingPerMemLvl = {}
    mem_hierarchy = accelerator.memory_hierarchy

    for layer_op in memory_operand_links.layer_operands:
        mem_op = memory_operand_links.layer_to_mem_op(layer_op)
        usm_copy = user_spatial_mapping.copy()
        mapping_per_mem_lvl[layer_op] = []
        memory_levels = mem_hierarchy.get_memory_levels(mem_op)

        for memory_level in memory_levels:
            spatial_mapping_lvl_dict: dict[LayerDim, int | float] = {}
            served_dimensions = memory_level.served_dimensions
            for oa_dim in served_dimensions:
                if oa_dim in usm_copy:
                    for layer_dim, unrolling in usm_copy[oa_dim].items():
                        if layer_dim in spatial_mapping_lvl_dict:
                            spatial_mapping_lvl_dict[layer_dim] *= unrolling
                        else:
                            spatial_mapping_lvl_dict[layer_dim] = unrolling
                    del usm_copy.data[oa_dim]
            mapping_per_mem_lvl[layer_op].append(list(spatial_mapping_lvl_dict.items()))

        # Top-level spillover for any remaining OA dims
        top_dict: dict[LayerDim, int | float] = {}
        for oa_dim, mapping_single_oa_dim in usm_copy.items():
            for layer_dim, unrolling in mapping_single_oa_dim.items():
                if layer_dim not in top_dict:
                    top_dict[layer_dim] = unrolling
                else:
                    top_dict[layer_dim] *= unrolling
        mapping_per_mem_lvl[layer_op].append(list(top_dict.items()))

    return mapping_per_mem_lvl


# ---------------------------------------------------------------------------
# LayerNode construction
# ---------------------------------------------------------------------------

def _build_layer_node(
    problem_dims: dict[int, int],
    template_bounds: tuple[int, ...],
    element_bytes: Sequence[int],
) -> LayerNode:
    """Construct a ZigZag LayerNode for a GEMM workload."""
    dim_sizes = LayerDimSizes({_dim(idx): size for idx, size in problem_dims.items()})

    # Precision in bits
    i_prec = element_bytes[0] * 8 if len(element_bytes) > 0 else 8
    w_prec = element_bytes[1] * 8 if len(element_bytes) > 1 else 8
    o_prec = element_bytes[2] * 8 if len(element_bytes) > 2 else 32

    precision = LayerOperandPrecision({
        _I: i_prec,
        _W: w_prec,
        _O: o_prec,
        Constants.FINAL_OUTPUT_LAYER_OP: o_prec,
    })

    node_attr = LayerNodeAttributes(
        layer_type="Gemm",
        equation=LayerEquation(GEMM_EQUATION),
        layer_dim_sizes=dim_sizes,
        operand_precision=precision,
        dimension_relations=[],
        padding=LayerPadding.empty(),
        constant_operands=[_W],
        input_operand_source={_I: 0, _W: 0},
        pr_layer_dim_sizes=None,
    )

    # Spatial mapping for the LayerNode (high-level, OADimension-based)
    oa_mapping: dict[OADimension, MappingSingleOADim] = {}
    for dim_idx, bound in enumerate(template_bounds):
        if bound > 1:
            oa_dim = OADimension(DIM_NAMES[dim_idx])
            oa_mapping[oa_dim] = MappingSingleOADim({_dim(dim_idx): bound})

    mapping_attr = MappingAttributes(
        spatial_mapping=SpatialMapping(oa_mapping),
        spatial_mapping_hint=SpatialMappingHint.empty(),
        memory_operand_links=MemoryOperandLinks({
            _I: _MEM_I1,
            _W: _MEM_I2,
            _O: _MEM_O,
        }),
        temporal_ordering=LayerTemporalOrdering.empty(),
    )

    return LayerNode(
        layer_id=0,
        node_name="snaxc_gemm",
        node_attr=node_attr,
        mapping_attr=mapping_attr,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def zigzag_cost_of_tiling(
    tiling: list[tuple[int, int, bool]],
    operand_descs: Sequence[OperandDescriptor],
    invariance_map: list[set[int]],
    *,
    template_bounds: tuple[int, ...] = (8, 8, 8),
    num_banks: int = 32,
    hw_yaml: str | None = None,
) -> float:
    """Evaluate a tiling using ZigZag's cost model and return the total cycle count.

    Parameters
    ----------
    tiling : list of (template_dim_idx, tile_size, is_critical), inner→outer.
    operand_descs : per-operand OperandDescriptor (from snaxc).
    invariance_map : per-operand set of template dim indices the operand is
        invariant to.
    template_bounds : spatial unrolling per template dimension.
    num_banks : number of TCDM banks (unused by ZigZag, kept for API compat).
    hw_yaml : path to the ZigZag hardware YAML.  Defaults to
        ``zigzag/inputs/hardware/gemm_l1_l3_fixed_cache.yaml``.

    Returns
    -------
    float : total execution cycles (``latency_total2`` from ZigZag).
    """
    if hw_yaml is None:
        hw_yaml = os.path.normpath(_DEFAULT_HW_YAML)

    # 1. Load hardware
    accelerator = _load_accelerator(hw_yaml)

    # 2. Truncate to the 3 GEMM operands (snaxc may pass more for QMAC)
    inv_map = invariance_map[:3] if len(invariance_map) >= 3 else invariance_map
    elem_bytes = [operand_descs[i].element_bytes for i in range(min(3, len(operand_descs)))]

    # 3. Reconstruct problem dimensions from tiling + spatial bounds
    problem_dims: dict[int, int] = {}
    for dim_idx in range(len(template_bounds)):
        total = template_bounds[dim_idx]
        for t_dim, t_size, _ in tiling:
            if t_dim == dim_idx:
                total *= t_size
        problem_dims[dim_idx] = total

    # 4. Build the LayerNode
    layer = _build_layer_node(problem_dims, template_bounds, elem_bytes)

    # 5. Initialize the high-level spatial mapping's OA dimension sizes
    #    (required by SpatialMappingConversionStage logic)
    oa_dim_sizes = {
        OADimension(DIM_NAMES[i]): template_bounds[i]
        for i in range(len(template_bounds))
    }
    layer.spatial_mapping.initialize_oa_dims(oa_dim_sizes)

    # 6. Convert spatial mapping to internal per-mem-level format using
    #    ZigZag's own conversion stage
    conversion = SpatialMappingConversionStage.__new__(SpatialMappingConversionStage)
    conversion.layer = layer
    conversion.accelerator = accelerator
    conversion.memory_operand_links = layer.memory_operand_links
    conversion.user_spatial_mapping = layer.spatial_mapping
    conversion.oa_dim_sizes = oa_dim_sizes
    spatial_mapping_internal, spatial_mapping_int = conversion.convert_user_spatial_mapping(
        layer.spatial_mapping
    )

    # 7. Determine temporal mapping levels per operand (= arch_level - 1)
    num_temporal_levels: dict[LayerOperand, int] = {
        op: spatial_mapping_internal.arch_level[op] - 1
        for op in [_I, _W, _O]
    }

    # 8. Build temporal mapping
    temporal_dict = _split_tiling_into_levels(tiling, inv_map, num_temporal_levels)
    temporal_mapping = TemporalMapping(temporal_dict, layer, TemporalMappingType.UNEVEN)

    # 9. Run ZigZag cost model
    cme = CostModelEvaluation(
        accelerator=accelerator,
        layer=layer,
        spatial_mapping=spatial_mapping_internal,
        spatial_mapping_int=spatial_mapping_int,
        temporal_mapping=temporal_mapping,
    )

    return cme.latency_total2
