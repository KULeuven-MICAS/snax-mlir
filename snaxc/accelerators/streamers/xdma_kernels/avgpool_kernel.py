from collections.abc import Sequence

from xdsl.dialects.builtin import IntegerAttr, i8
from xdsl.ir import Operation, ParametrizedAttribute, SSAValue
from xdsl.ir.affine import AffineMap

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.streamers.extensions.avgpool_extension import AvgPoolExtension
from snaxc.accelerators.streamers.extensions.rescale_extension import (
    RescaleDownExtension,
)
from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension
from snaxc.accelerators.streamers.streamers import (
    Streamer,
    StreamerConfiguration,
    StreamerType,
)
from snaxc.accelerators.streamers.xdma_kernels.xdma_kernel import XDMAKernel
from snaxc.dialects import dart, kernel
from snaxc.ir.dart.access_pattern import Template, TemplatePattern


class AvgPoolXDMA(XDMAKernel):
    """
    Snax XDMA AvgPool Kernel
    This kernel is used to perform average pooling operations on the XDMA core.
    """

    required_extensions: Sequence[StreamerExtension] = (
        AvgPoolExtension(),
        RescaleDownExtension(),
    )
    supported_kernel = SupportedKernel(kernel.AvgPoolOp, [i8, i8, i8])

    def get_csr_values(self, op: kernel.KernelOp) -> Sequence[Sequence[int]]:
        """
        Returns the CSR values for the MaxPool extension.
        This method should be implemented to provide the specific CSR values needed for the MaxPool extension.
        """
        assert isinstance(op.attributes["kernel_size"], IntegerAttr)
        add_to_long_csr = [
            op.attributes["kernel_size"].value.data,
        ]
        rescale_down_csr = [
            0,
            (2**25) // op.attributes["kernel_size"].value.data,
            0,
            25
        ]  # TODO: PLACEHOLDER VALUES: figure out how these are actually determined

        return [add_to_long_csr, rescale_down_csr]

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
        new_inputs = [op.inputs[0]]
        new_outputs: list[SSAValue] = list(op.outputs)

        snax_stride_patterns = list(snax_stride_patterns)

        return (
            new_inputs,
            new_outputs,
            [snax_stride_patterns[0], snax_stride_patterns[-1]],
            [],
        )
