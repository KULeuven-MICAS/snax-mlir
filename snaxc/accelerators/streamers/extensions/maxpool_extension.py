from collections.abc import Sequence

from xdsl.dialects.builtin import i8
from xdsl.ir import Operation, ParametrizedAttribute, SSAValue
from xdsl.ir.affine import AffineMap

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension
from snaxc.accelerators.streamers.streamers import (
    Streamer,
    StreamerConfiguration,
    StreamerType,
)
from snaxc.dialects import dart, kernel
from snaxc.ir.dart.access_pattern import Template, TemplatePattern


class MaxPoolExtension(StreamerExtension):
    """
    Snax XDMA MaxPool Extension
    This extension is used to perform max pooling operations on the XDMA core.
    """

    name = "maxpool_ext"
    supported_kernel = SupportedKernel(kernel.MaxPoolOp, [i8, i8, i8])
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
        template = [
            AffineMap.from_callable(lambda y: (y,)),
            AffineMap.from_callable(lambda _: tuple()),
            AffineMap.from_callable(lambda y: (y,)),
        ]
        template_bounds = (64,)
        return Template(TemplatePattern(template_bounds, tp) for tp in template)

    def get_streamers(
        self, streamer_config: StreamerConfiguration
    ) -> Sequence[Streamer]:
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
        Set the stride patterns for the MaxPool operation.
        The Middle pattern is removed as it is not needed for MaxPool.
        """
        snax_stride_patterns = list(snax_stride_patterns)

        return (
            op.inputs,
            op.outputs,
            [snax_stride_patterns[0], snax_stride_patterns[-1]],
            [],
        )
