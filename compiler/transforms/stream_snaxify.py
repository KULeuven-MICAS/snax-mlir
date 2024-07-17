from dataclasses import dataclass

from xdsl.dialects import builtin, memref, memref_stream
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import MLContext, Region
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable

from compiler.dialects import snax_stream
from compiler.dialects.accfg import AcceleratorOp
from compiler.dialects.snax import StreamerConfigurationAttr


@dataclass
class MemrefStreamToSnaxPattern(RewritePattern):
    """
    This pattern converts memref_stream streaming
    region ops into snax_stream streaming region ops.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref_stream.StreamingRegionOp, rewriter: PatternRewriter):
        """
        Current restrictions:
            - only allow default memref layouts
        """

        # only memref stream ops dispatched to an accelerator:
        if "accelerator" not in op.attributes:
            return

        accelerator_str = op.attributes["accelerator"]
        assert isinstance(accelerator_str, StringAttr)

        shaped_operands = [*op.inputs, *op.outputs]

        # make sure there are as much shaped operands as access patterns
        assert len(shaped_operands) == len(op.patterns)

        # make sure the operands are memrefs with default layout
        for memref_operand in shaped_operands:
            if not isinstance(memref_operand.type, builtin.MemRefType):
                return
            if not isinstance(memref_operand.type.layout, builtin.NoneAttr):
                return

        # construct the strided patterns for SNAX Streamers

        # get the affine mapping from data to memory:
        data_mem_maps = [AffineMap.identity(1) for _ in shaped_operands]

        # combine the maps access -> data and data -> memory to access -> memory
        access_mem_maps = [
            data_mem_map.compose(memref_stride_pattern.index_map.data)
            for data_mem_map, memref_stride_pattern in zip(data_mem_maps, op.patterns.data)
        ]

        # now we have a mapping from access to memory, this can be used to program the streamers
        # first, find the streamer config
        module_op = op
        while module_op and not isinstance(module_op, ModuleOp):
            module_op = module_op.parent_op()
        if not module_op:
            raise RuntimeError("Module Op not found")

        trait = module_op.get_trait(SymbolTable)
        assert trait is not None
        acc_op = trait.lookup_symbol(module_op, accelerator_str)

        if not isinstance(acc_op, AcceleratorOp):
            raise RuntimeError("AcceleratorOp not found!")

        streamer_config = acc_op.attributes["streamer_config"]
        if not isinstance(streamer_config, StreamerConfigurationAttr):
            raise RuntimeError("Streamer interface not found for given accelerator op")

        # small function to generate a list of n zeros with the i-th element 1
        def generate_one_list(n: int, i: int):
            return [1 if j == i else 0 for j in range(n)]


        snax_stride_patterns = []
        for access_mem_map, memref_stride_pattern in zip(access_mem_maps, op.patterns.data):
            # find accelerator in attribute, look up accelerator op in the ir.
            # get the streamer access patterns from the interface attribute of the accfg op
            # map the access pattern to stride pattern spatial first, then temporal

            temporal_strides = []
            spatial_strides = []
            upper_bounds = []
            access_mem_map_dim = access_mem_map.num_dims

            # do not consider symbols yet
            if access_mem_map.num_symbols != 0:
                raise RuntimeError("Access patterns with symbols are not supported yet.")

            # first, fill up the spatial strides
            for i in reversed(
                range(
                    streamer_config.data.temporal_dim(),
                    streamer_config.data.temporal_dim() + streamer_config.data.spatial_dim(),
                )
            ):
                stride = access_mem_map.eval(generate_one_list(access_mem_map_dim, i), ())
                spatial_strides.append(stride[0])

            # then, fill up the temporal strides
            for i in range(streamer_config.data.temporal_dim()):
                stride = access_mem_map.eval(generate_one_list(access_mem_map_dim, i), ())
                temporal_strides.append(stride[0])
                upper_bounds.append(memref_stride_pattern.ub.data[i].data)

            # create the stride pattern
            snax_stride_pattern = snax_stream.StridePattern(
                upper_bounds=upper_bounds,
                temporal_strides=temporal_strides,
                spatial_strides=spatial_strides,
            )
            snax_stride_patterns.append(snax_stride_pattern)


        # get base addresses of the streaming region ops
        # TODO: generalize and fix for offsets

        new_inputs = [
            memref.ExtractAlignedPointerAsIndexOp.get(input)
            for input in op.inputs
        ]

        new_outputs = [
            memref.ExtractAlignedPointerAsIndexOp.get(output)
            for output in op.outputs
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
    """
    A pass to convert memref_stream operations to snax stream. This
    boils down to combining the data access patterns of a memref_stream
    op (operation -> data), with a certain data layout (data -> memory)
    and realizing this by a snax_stream access pattern
    (operation -> memory), which is then realized by the Streamers.
    """

    name = "stream-snaxify"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(MemrefStreamToSnaxPattern()).rewrite_module(op)
