from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, memref, memref_stream
from xdsl.dialects.builtin import MemRefType, StringAttr
from xdsl.ir import Operation
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

        snax_stride_patterns: list[snax_stream.StridePattern] = []

        # small function to generate a list of n zeros with the i-th element 1
        # for example n = 4, i = 1  -> [0, 1, 0, 0]
        def generate_one_list(n: int, i: int):
            return [1 if j == i else 0 for j in range(n)]

        # Do this for every operand:
        for operand in range(len(op.operands)):
            # Mapping from data to memory:
            assert isinstance(memref_type := op.operands[operand].type, MemRefType)

            # Mapping from data to memory:
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

            # Get the streamer
            # FIXME: this is code copied from schedule memref linalg,
            # move this to accelerator definitions to avoid code duplication
            acc_str = acc_op.name_prop.root_reference.data
            template_bounds: tuple[int | None, ...] = ()
            if acc_str == "snax_alu":
                template_bounds = (None, 4)
            elif acc_str == "snax_gemm":
                template_bounds = (None, None, None, 8, 8, 8)
            elif acc_str == "snax_gemmx":
                if len(op.inputs) > 1:
                    # gemm
                    template_bounds = (None, None, None, 8, 8, 8)
                else:
                    # simd only
                    template_bounds = (None, None, 8, 8)

            # Create iterator for all dimensions of the access_mem_map that returns (stride, bound)
            access_iter = iter(
                (
                    access_mem_map.eval(
                        generate_one_list(access_mem_map.num_dims, i), ()
                    )[0],
                    op.patterns.data[operand].ub.data[i].value.data,
                )
                for i in reversed(range(access_mem_map.num_dims))
            )

            # Fetch the first stride
            stride, bound = next(access_iter)

            temporal_strides: list[int] = []
            spatial_strides: list[int] = []
            upper_bounds: list[int] = []

            # fill up all spatial strides
            for _ in [x for x in template_bounds if x is not None]:
                # FIXME: provide more general solution for new spatial streamer_config
                # configuration, this works in all current cases and layouts but is far from generally correct.
                spatial_strides = [8]
                stride, bound = next(access_iter, (None, None))

            # remaining are temporal strides
            while stride is not None and bound is not None:
                temporal_strides.append(stride)
                upper_bounds.append(bound)
                stride, bound = next(access_iter, (None, None))

            # create the stride pattern for this operand
            snax_stride_pattern = snax_stream.StridePattern(
                upper_bounds=upper_bounds,
                temporal_strides=temporal_strides,
                spatial_strides=spatial_strides,
            )
            snax_stride_patterns.append(snax_stride_pattern)

        # get base addresses of the streaming region ops
        # TODO: generalize and fix for offsets

        new_inputs: list[Operation] = [
            memref.ExtractAlignedPointerAsIndexOp.get(input) for input in op.inputs
        ]
        new_outputs = [
            memref.ExtractAlignedPointerAsIndexOp.get(output) for output in op.outputs
        ]

        # TODO: what is still required is a better system for the unused operands
        # of snax_gemmx / other accelerators. this now fills in empty/zero patterns for the unused operands.

        if acc_op.name_prop.root_reference.data == "snax_gemmx":
            empty_pattern = snax_stream.StridePattern(
                upper_bounds=[0] * 3, temporal_strides=[0] * 3, spatial_strides=[0]
            )
            if len(snax_stride_patterns) == 3:
                # gemm

                # for a gemm, the 8-bit output port D8 and the bias in put C
                # are unused, so we create empty patterns for them here.

                # insert empty patterns for D8 and zero pattern for C
                snax_stride_patterns.insert(2, empty_pattern)
                new_inputs.append(
                    memref.ExtractAlignedPointerAsIndexOp.get(op.inputs[-1])
                )

                # insert zero pattern for C, using the same pattern as D32 but pointing to zero
                # this way, the bias used by the gemm is just a bunch of zeros
                snax_stride_patterns.insert(
                    3,
                    snax_stream.StridePattern(
                        upper_bounds=snax_stride_patterns[3].upper_bounds,
                        temporal_strides=[0] * 3,
                        spatial_strides=[8],
                    ),
                )

                # point C to c0
                new_inputs.append(
                    # zero pointer will generate 0 values
                    arith.Constant.from_int_and_width(0, builtin.IndexType())
                )

            else:
                # simd
                # to calculate only simd, we calculate the result
                # of D8 = rescale(AxB + C)
                # create zero patterns for A and B such that D8 = rescale(C)
                # create empty pattern for D32
                # do not use new outputs
                new_inputs.append(new_outputs.pop())

                zero_pattern = snax_stream.StridePattern(
                    upper_bounds=snax_stride_patterns[0].upper_bounds,
                    temporal_strides=[0] * len(snax_stride_patterns[0].upper_bounds),
                    spatial_strides=[8],
                )

                # read zeros from tcdm (must make sure there are zeros at these addresses)
                # in the new streamer this can be fixed with byte masking
                snax_stride_patterns.insert(0, zero_pattern)
                new_inputs.insert(
                    0,
                    # zero pointer will generate 0 values
                    arith.Constant.from_int_and_width(0, builtin.IndexType()),
                )
                snax_stride_patterns.insert(1, zero_pattern)
                new_inputs.insert(
                    1,
                    # zero pointer will generate 0 values
                    arith.Constant.from_int_and_width(0, builtin.IndexType()),
                )

                # flip D8 and C such that they are in the right order
                snax_stride_patterns.append(snax_stride_patterns.pop(2))
                new_inputs.append(new_inputs.pop(2))

                # empty pattern for D32
                snax_stride_patterns.append(empty_pattern)
                # dummy base pointer for D32
                new_inputs.append(
                    memref.ExtractAlignedPointerAsIndexOp.get(op.inputs[-1])
                )

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
