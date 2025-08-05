from abc import ABC, abstractmethod
from collections.abc import Sequence

from xdsl.ir import Operation, ParametrizedAttribute, SSAValue
from xdsl.ir.affine import AffineMap

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.streamers.streamers import (
    Streamer,
    StreamerConfiguration,
    StreamerOpts,
)
from snaxc.dialects import dart, kernel
from snaxc.dialects.kernel import KernelOp
from snaxc.ir.dart.access_pattern import Template, TemplatePattern


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

    def get_template(self, op: kernel.KernelOp) -> Template:
        """
        Returns the template for this DMA extension.
        """
        template = [AffineMap.from_callable(lambda y: (y,))] * 2
        template_bounds = (16,)
        return Template(TemplatePattern(template_bounds, tp) for tp in template)

    def get_streamers(self, streamer_config: StreamerConfiguration) -> Sequence[Streamer]:
        """
        Returns the streamers for this DMA extension.
        """
        return streamer_config.streamers

    def set_stride_patterns(
        self,
        op: dart.AccessPatternOp,
        kernel_op: KernelOp,
        snax_stride_patterns: Sequence[ParametrizedAttribute],
    ) -> tuple[
        Sequence[SSAValue],
        Sequence[SSAValue],
        Sequence[ParametrizedAttribute],
        Sequence[Operation],
    ]:
        """
        Sets the stride patterns for the given access pattern operation.
        This method should be implemented to provide the specific stride patterns needed for the DMA extension.
        """
        return op.inputs, op.outputs, snax_stride_patterns, []
