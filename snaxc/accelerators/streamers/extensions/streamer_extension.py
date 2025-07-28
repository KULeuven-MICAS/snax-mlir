from abc import ABC, abstractmethod
from collections.abc import Sequence

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.streamers.streamers import StreamerOpts
from snaxc.dialects.kernel import KernelOp


class StreamerExtension(StreamerOpts, ABC):
    """
    Abstract base class for DMA extensions.
    This class defines the interface for DMA extensions that can be used
    with accelerators.
    """

    name: str
    supported_kernel: SupportedKernel | None
    csr_length: int

    @abstractmethod
    def get_dma_extension_name(self) -> str:
        """
        Returns the name of the DMA extension.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_dma_extension_kernel(self) -> SupportedKernel | None:
        """
        Returns the supported kernel for this DMA extension.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_csr_values(self, op: KernelOp) -> Sequence[int]:
        """
        Returns the CSR values for this DMA extension.
        This method should be implemented by subclasses to provide
        the specific CSR values needed for the DMA extension.
        length of the CSR values should match csr_length.
        """
        raise NotImplementedError()
