from abc import ABC, abstractmethod
from collections.abc import Sequence

from xdsl.ir import Operation, SSAValue

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.streamers.streamers import (
    Streamer,
    StreamerConfiguration,
    StreamerOpts,
)
from snaxc.dialects import dart
from snaxc.dialects.kernel import KernelOp
from snaxc.dialects.snax_stream import StridePattern
from snaxc.ir.dart.access_pattern import Template


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

    @abstractmethod
    def get_template(self, op: KernelOp) -> Template:
        """
        Returns the template for this DMA extension.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_streamers(self, streamer_config: StreamerConfiguration) -> Sequence[Streamer]:
        """
        Returns the streamers for this DMA extension.
        This method should be implemented to provide the specific streamers needed for the Add extension.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_stride_patterns(
        self,
        op: dart.AccessPatternOp,
        kernel_op: KernelOp,
        snax_stride_patterns: Sequence[StridePattern],
    ) -> tuple[
        Sequence[SSAValue],
        Sequence[SSAValue],
        Sequence[StridePattern],
        Sequence[Operation],
    ]:
        """
        Sets the stride patterns for the given access pattern operation.
        This method should be implemented to provide the specific stride patterns needed for the DMA extension.
        """
        raise NotImplementedError()
