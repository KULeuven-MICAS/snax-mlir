from abc import ABC, abstractmethod
from collections.abc import Sequence

from xdsl.ir import Operation, ParametrizedAttribute, SSAValue
from xdsl.ir.affine import AffineMap

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.streamers.extensions import StreamerExtension
from snaxc.accelerators.streamers.streamers import (
    Streamer,
    StreamerConfiguration,
)
from snaxc.dialects import dart, kernel
from snaxc.dialects.kernel import KernelOp
from snaxc.ir.dart.access_pattern import Template, TemplatePattern


class XDMAKernel(ABC):
    """
    Abstract base class for XDMA kernels.
    This class defines the interface for kernels that can be used
    with accelerators.
    """

    # Required extensions must be in the order they are applied
    required_extensions: Sequence[StreamerExtension]

    supported_kernel: SupportedKernel

    @abstractmethod
    def get_csr_values(self, op: KernelOp) -> Sequence[Sequence[int]]:
        """
        Returns the CSR values for this all extensions of this kernel.
        This method should be implemented by subclasses to provide
        the specific CSR values needed for the DMA extension.
        length of the CSR values should match csr_length.
        """
        raise NotImplementedError()

    def get_template(self, op: kernel.KernelOp) -> Template:
        """
        Returns the template for this kernel.
        """
        template = [AffineMap.from_callable(lambda y: (y,))] * 2
        template_bounds = (16,)
        return Template(TemplatePattern(template_bounds, tp) for tp in template)

    def get_streamers(
        self, streamer_config: StreamerConfiguration
    ) -> Sequence[Streamer]:
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
