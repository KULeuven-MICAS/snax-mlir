from collections.abc import Sequence

from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import AffineMap

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension
from snaxc.accelerators.streamers.streamers import Streamer, StreamerConfiguration
from snaxc.dialects import dart, kernel
from snaxc.dialects.snax_stream import StridePattern
from snaxc.ir.dart.access_pattern import Template, TemplatePattern


class MaxPoolExtension(StreamerExtension):
    """
    Snax XDMA MaxPool Extension
    This extension is used to perform max pooling operations on the XDMA core.
    """

    name = "maxpool_ext"
    supported_kernel = None  # TODO: Select correct kernel
    csr_length = 1

    def get_dma_extension_name(self) -> str:
        return self.name

    def get_dma_extension_kernel(self) -> SupportedKernel | None:
        return self.supported_kernel

    def get_csr_values(self, op: kernel.KernelOp) -> Sequence[int]:
        """
        Returns the CSR values for the MaxPool extension.
        This method should be implemented to provide the specific CSR values needed for the MaxPool extension.
        """
        # Example implementation, replace with actual logic
        return [1]  # TODO: Replace with actual CSR values for MaxPool

    def get_template(self, op: kernel.KernelOp) -> Template:
        """
        Returns the template for this DMA extension.
        """
        template = [AffineMap.from_callable(lambda y: (y,))] * 2  # TODO: check if this is correct
        template_bounds = (16,)
        return Template(TemplatePattern(template_bounds, tp) for tp in template)

    def get_streamers(self, streamer_config: StreamerConfiguration) -> Sequence[Streamer]:
        """
        Returns the streamers for this DMA extension.
        This method should be implemented to provide the specific streamers needed for the Add extension.
        """
        return [
            streamer_config.streamers[0],
            streamer_config.streamers[1],
        ]

    def set_stride_patterns(
        self,
        op: dart.AccessPatternOp,
        kernel_op: kernel.KernelOp,
        snax_stride_patterns: Sequence[StridePattern],
    ) -> tuple[
        Sequence[SSAValue],
        Sequence[SSAValue],
        Sequence[StridePattern],
        Sequence[Operation],
    ]:
        snax_stride_patterns = list(snax_stride_patterns)
        new_inputs = [op.inputs[0]]
        new_outputs: list[SSAValue] = list(op.outputs)

        pattern = snax_stride_patterns[0]
        new_stride_pattern = StridePattern(
            [x.data for x in pattern.upper_bounds],
            [x.data for x in pattern.temporal_strides],  # TODO: check if this is correct
            pattern.spatial_strides,
        )

        new_stride_patterns = [new_stride_pattern, snax_stride_patterns[-1]]

        return new_inputs, new_outputs, new_stride_patterns, []
