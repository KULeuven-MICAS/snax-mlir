from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import builtin, memref, memref_stream
from xdsl.dialects.builtin import FixedBitwidthType, MemRefType, StringAttr
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.accelerators import find_accelerator_op
from compiler.dialects import snax_stream
from compiler.dialects.snax import StreamerConfigurationAttr
from compiler.dialects.tsl import TiledStridedLayoutAttr


@dataclass
class HoistAcceleratorAttribute(RewritePattern):
    """
    A pattern to hoist the library call from within a memref
    streaming region as an attribute of the op.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref_stream.StreamingRegionOp, _):
        # attribute already assigned to op
        if getattr(op.attributes, "accelerator", None):
            return

        accelerator = None

        for inner_op in op.body.walk():
            # look for memref_stream.generic with library call
            if not isinstance(inner_op, memref_stream.GenericOp):
                continue

            if not inner_op.library_call:
                continue

            if accelerator and inner_op.library_call != accelerator:
                raise RuntimeError(
                    "multiple different accelerator dispatches found in a single memref_stream.streaming_region op"
                )

            accelerator = inner_op.library_call.data

        if accelerator:
            op.attributes["accelerator"] = StringAttr(accelerator)


@dataclass
class MemrefStreamToSnaxPattern(RewritePattern):
    """
    A pass to convert memref_stream operations to snax stream.

    This boils down to combining the data access patterns of a memref_stream op (operation -> data),
    with a certain data layout: an affine map from (data -> memory) into a mapping (operation -> memory).

    This takes the form of a snax_stream access pattern, mapping (operation -> memory)
    which, in hardware, is  realized by the Streamers.

    Current restrictions:
        We are only handling default memory layouts for now (NoneAttr)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.StreamingRegionOp, rewriter: PatternRewriter
    ):
        # Compliance checks:

        # Handle only memref stream ops dispatched to an accelerator:
        if "accelerator" not in op.attributes:
            return

        # Go and fetch the accelerator op
        assert isinstance((accelerator_str := op.attributes["accelerator"]), StringAttr)
        acc_op = find_accelerator_op(op, accelerator_str)

        if not acc_op:
            raise RuntimeError("AcceleratorOp not found!")

        if "streamer_config" not in acc_op.attributes:
            raise RuntimeError("Streamer interface not found for given accelerator op")
        streamer_config = acc_op.attributes["streamer_config"]
        assert isinstance(streamer_config, StreamerConfigurationAttr)

        # Make sure the operands are memrefs
        for memref_operand in op.operands:
            if not isinstance(memref_operand.type, builtin.MemRefType):
                return

        # We are now ready to convert the stream access patterns into snax stride patterns
        # construct the strided patterns for SNAX Streamers

        snax_stride_patterns = []

        # small function to generate a list of n zeros with the i-th element 1
        # for example n = 4, i = 1  -> [0, 1, 0, 0]
        def generate_one_list(n: int, i: int):
            return [1 if j == i else 0 for j in range(n)]

        # Do this for every operand:
        for operand in range(len(op.operands)):
            # Mapping from data to memory:
            assert isinstance(memref_type := op.operands[operand].type, MemRefType)

            #TODO: fix element offset in tsl to avoid this shit
            if isinstance(memref_type.layout, TiledStridedLayoutAttr):
                data_mem_map : AffineMap = memref_type.get_affine_map()
            else:
                data_mem_map: AffineMap = memref_type.get_affine_map_in_bytes()

            # Mapping from access to data:
            access_data_map: AffineMap = op.patterns.data[operand].index_map.data

            # Mapping from access to memory:
            access_mem_map: AffineMap = data_mem_map.compose(access_data_map)

            # Make sure no symbols are used (not supported yet)
            if access_mem_map.num_symbols != 0:
                raise RuntimeError(
                    "Access patterns with symbols are not supported yet."
                )

            temp_dim = streamer_config.data.temporal_dim()
            spat_dim = streamer_config.data.spatial_dim()

            # extremely dirty fix:
            if spat_dim == 2:
                spat_dim = 3

            temporal_strides = []
            spatial_strides = []
            upper_bounds = []


            # First fill up the spatial strides, then temporal strides, back to front
            for i in reversed(range(access_mem_map.num_dims)):
                stride = access_mem_map.eval(
                    generate_one_list(access_mem_map.num_dims, i), ()
                )
                if len(spatial_strides) < spat_dim:
                    # keep filling up spatial strides
                    spatial_strides.append(stride[0])
                else:
                    # filling up the temporal strides
                    temporal_strides.append(stride[0])
                    # have to set upper bounds for spatial strides
                    upper_bounds.append(op.patterns.data[operand].ub.data[i].value)

            # delete all zeros from spatial strides
            spatial_strides = [x for x in spatial_strides if x]

            # create the stride pattern for this operand
            snax_stride_pattern = snax_stream.StridePattern(
                upper_bounds=upper_bounds,
                temporal_strides=temporal_strides,
                spatial_strides=spatial_strides,
            )
            snax_stride_patterns.append(snax_stride_pattern)

        # get base addresses of the streaming region ops
        # TODO: generalize and fix for offsets

        new_inputs = [
            memref.ExtractAlignedPointerAsIndexOp.get(input) for input in op.inputs
        ]
        new_outputs = [
            memref.ExtractAlignedPointerAsIndexOp.get(output) for output in op.outputs
        ]

        # now create snax_streaming region op
        new_op = snax_stream.StreamingRegionOp(
            inputs=new_inputs,
            outputs=new_outputs,
            stride_patterns=snax_stride_patterns,
            accelerator=accelerator_str,
            body=rewriter.move_region_contents_to_new_regions(op.body),
        )

        rewriter.replace_matched_op([*new_inputs, *new_outputs, new_op], new_op.results)


@dataclass(frozen=True)
class StreamSnaxify(ModulePass):
    name = "stream-snaxify"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [HoistAcceleratorAttribute(), MemrefStreamToSnaxPattern()]
            )
        ).rewrite_module(op)
