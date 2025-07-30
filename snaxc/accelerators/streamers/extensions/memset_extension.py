from collections.abc import Sequence

from xdsl.ir.affine import AffineMap

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension
from snaxc.accelerators.streamers.streamers import Streamer, StreamerConfiguration
from snaxc.dialects import kernel
from snaxc.ir.dart.access_pattern import Template, TemplatePattern


class MemSetExtension(StreamerExtension):
    """
    Snax XDMA MemSet Extension
    This extension is used to perform memory set operations on the XDMA core.
    """

    name = "memset_ext"
    supported_kernel = None  # TODO: Select correct kernel
    csr_length = 1

    def get_dma_extension_name(self) -> str:
        return self.name

    def get_dma_extension_kernel(self) -> SupportedKernel | None:
        return self.supported_kernel

    def get_csr_values(self, op: kernel.KernelOp) -> Sequence[int]:
        """
        Returns the CSR values for the MemSet extension.
        This method should be implemented to provide the specific CSR values needed for the MemSet extension.
        """
        # Example implementation, replace with actual logic
        return [0]  # TODO: Replace with actual CSR values for MemSet

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
