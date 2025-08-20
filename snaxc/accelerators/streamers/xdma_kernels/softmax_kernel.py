import math
from collections.abc import Sequence

from xdsl.dialects.builtin import i8, i32
from xdsl.ir import Operation, ParametrizedAttribute, SSAValue
from xdsl.ir.affine import AffineMap

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.streamers.extensions.softmax_extension import SoftMaxExtension
from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension
from snaxc.accelerators.streamers.streamers import (
    Streamer,
    StreamerConfiguration,
    StreamerType,
)
from snaxc.accelerators.streamers.xdma_kernels.xdma_kernel import XDMAKernel
from snaxc.dialects import dart, kernel
from snaxc.dialects.snax_stream import StridePattern
from snaxc.ir.dart.access_pattern import Template, TemplatePattern


class SoftMaxXDMA(XDMAKernel):
    """
    Snax XDMA SoftMax Extension
    This extension is used to perform softmax operations on the XDMA core.
    """

    required_extensions: Sequence[StreamerExtension] = (SoftMaxExtension(),)
    supported_kernel = SupportedKernel(kernel.SoftMaxOp, [i32, i8, i32])

    def get_csr_values(self, op: kernel.KernelOp) -> Sequence[Sequence[int]]:
        """
        Returns the CSR values for the SoftMax extension.
        This method should be implemented to provide the specific CSR values needed for the SoftMax extension.
        """
        # Example implementation, replace with actual logic
        assert isinstance(op, kernel.SoftMaxOp), "Operation must be a SoftMaxOp"

        scaling_factor = op.scaling_factor.value.data

        scaled_ln2 = int(math.log(2) * scaling_factor)
        scaled_a = int(0.3585 * scaling_factor)
        scaled_b = int(1.353 * scaling_factor)
        scaled_c = int(0.344 * (scaling_factor**3)) >> math.floor(math.log2(scaling_factor) * 2)
        shift = math.floor(math.log2(scaling_factor) * 2)
        vector_length = op.vector_length.value.data

        return [[scaled_ln2, scaled_a, scaled_b, scaled_c, shift, vector_length]]

    def get_template(self, op: kernel.KernelOp) -> Template:
        template = [
            AffineMap.from_callable(lambda y: (y,)),
            AffineMap.from_callable(lambda _: tuple()),
            AffineMap.from_callable(lambda y: (y,)),
        ]
        template_bounds = (16,)
        return Template(TemplatePattern(template_bounds, tp) for tp in template)

    def get_streamers(self, streamer_config: StreamerConfiguration) -> Sequence[Streamer]:
        """
        Returns the streamers for this DMA extension.
        """
        return [
            streamer_config.streamers[0],
            Streamer(StreamerType.Reader, ("i",) * 6, [], []),
            streamer_config.streamers[1],
        ]

    def set_stride_patterns(
        self,
        op: dart.AccessPatternOp,
        kernel_op: kernel.KernelOp,
        snax_stride_patterns: Sequence[ParametrizedAttribute],
    ) -> tuple[
        Sequence[SSAValue],
        Sequence[SSAValue],
        Sequence[ParametrizedAttribute],
        Sequence[Operation],
    ]:
        """
        Set the stride patterns for the SoftMax operation.
        The Middle pattern is removed as it is not needed for SoftMax.
        """
        new_inputs = [op.inputs[0]]
        new_outputs: list[SSAValue] = list(op.outputs)

        snax_stride_patterns = list(snax_stride_patterns)

        assert isinstance(kernel_op, kernel.SoftMaxOp), "Operation must be a SoftMaxOp"
        vector_length = kernel_op.vector_length.value.data

        # Ensure the vector is the inner dimension of the pattern in the input
        input_snax_stride_pattern = snax_stride_patterns[0]
        assert isinstance(input_snax_stride_pattern, StridePattern)

        input_index = None
        input_bound_list = [bound.data for bound in input_snax_stride_pattern.upper_bounds]
        for i in range(len(input_snax_stride_pattern.upper_bounds)):
            if input_bound_list[i] == vector_length:
                input_index = i
                break
        assert input_index is not None, "Input stride pattern must contain vector length"

        # Bring the last upper bound and temporal stride to the front
        input_snax_stride_pattern = StridePattern(
            [bound.data for i, bound in enumerate(input_snax_stride_pattern.upper_bounds) if i == input_index]
            + [bound.data for i, bound in enumerate(input_snax_stride_pattern.upper_bounds) if i != input_index],
            [stride.data for i, stride in enumerate(input_snax_stride_pattern.temporal_strides) if i == input_index]
            + [stride.data for i, stride in enumerate(input_snax_stride_pattern.temporal_strides) if i != input_index],
            input_snax_stride_pattern.spatial_strides,
        )

        output_snax_stride_pattern = snax_stride_patterns[-1]
        assert isinstance(output_snax_stride_pattern, StridePattern)

        # Ensure the vector is the inner dimension of the pattern in the output
        output_index = None
        output_bound_list = [bound.data for bound in output_snax_stride_pattern.upper_bounds]
        for i in range(len(output_snax_stride_pattern.upper_bounds)):
            if output_bound_list[i] == vector_length:
                output_index = i
                break
        assert output_index is not None, "Output stride pattern must contain vector length"
        # Bring the last upper bound and temporal stride to the front
        output_snax_stride_pattern = StridePattern(
            [bound.data for i, bound in enumerate(output_snax_stride_pattern.upper_bounds) if i == output_index]
            + [bound.data for i, bound in enumerate(output_snax_stride_pattern.upper_bounds) if i != output_index],
            [stride.data for i, stride in enumerate(output_snax_stride_pattern.temporal_strides) if i == output_index]
            + [
                stride.data for i, stride in enumerate(output_snax_stride_pattern.temporal_strides) if i != output_index
            ],
            output_snax_stride_pattern.spatial_strides,
        )

        # Remove the triple repeat stride pattern from output (only needed for input)
        output_index = None
        output_bound_list = [bound.data for bound in output_snax_stride_pattern.upper_bounds]
        output_stride_list = [stride.data for stride in output_snax_stride_pattern.temporal_strides]
        # Find the index of the stride pattern that has a value of 3
        for i in range(len(output_bound_list)):
            if output_bound_list[i] == 3 and output_stride_list[i] == 0:
                output_index = i
                break
        assert output_index is not None, "Output stride pattern must contain triple repeat stride"

        output_snax_stride_pattern = StridePattern(
            [bound.data for i, bound in enumerate(output_snax_stride_pattern.upper_bounds) if i != output_index],
            [stride.data for i, stride in enumerate(output_snax_stride_pattern.temporal_strides) if i != output_index],
            output_snax_stride_pattern.spatial_strides,
        )

        return (
            new_inputs,
            new_outputs,
            [input_snax_stride_pattern, output_snax_stride_pattern],
            [],
        )
