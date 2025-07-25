from collections.abc import Sequence

from xdsl.dialects.builtin import i32

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension
from snaxc.dialects import kernel


class TransposeExtension(StreamerExtension):
    """
    Snax XDMA Transpose Extension
    This extension is used to perform transpose operations on the XDMA core.
    """

    name = "t"
    supported_kernel = SupportedKernel(kernel.MacOp, [i32, i32, i32])  # TODO: Select correct kernel
    csr_length = 1

    def get_dma_extension_name(self) -> str:
        return self.name

    def get_dma_extension_kernel(self) -> SupportedKernel:
        return self.supported_kernel

    def get_csr_values(self, op: kernel.KernelOp) -> Sequence[int]:
        """
        Returns the CSR values for the Transpose extension.
        This method should be implemented to provide the specific CSR values needed for the Transpose extension.
        """
        # Example implementation, replace with actual logic
        return [3]  # TODO: Replace with actual CSR values for Transpose
