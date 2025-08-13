from abc import ABC
from collections.abc import Sequence

from xdsl.dialects.builtin import i8, i32

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.streamers.extensions.rescale_extension import RescaleDownExtension, RescaleUpExtension
from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension
from snaxc.accelerators.streamers.xdma_kernels.xdma_kernel import XDMAKernel
from snaxc.dialects import kernel


class RescaleXDMA(XDMAKernel, ABC):
    """
    Snax XDMA Rescale Extension
    This extension is used to perform Rescale operations on the XDMA core.
    """

    supported_kernel: SupportedKernel
    required_extensions: Sequence[StreamerExtension]

    def get_csr_values(self, op: kernel.KernelOp) -> Sequence[Sequence[int]]:
        """
        Returns the CSR values for the Rescale Down extension.
        This method should be implemented to provide the specific CSR values needed for the Rescale Down extension.
        """
        # returns the number of input tensors in the rescale down op, currently limited to 2
        assert isinstance(op, kernel.RescaleOp), "Operation must be a RescaleOp"
        multiplier = op.multiplier.get_values()[0]
        shift = op.shift.get_values()[0]
        assert isinstance(multiplier, int), "Multiplier must be an integer"
        assert isinstance(shift, int), "Shift must be an integer"

        return [
            [
                op.input_zp.value.data,
                multiplier,
                op.output_zp.value.data,
                shift,
            ]
        ]


class RescaleDownXDMA(RescaleXDMA):
    """
    Snax XDMA Rescale Down Extension
    """

    required_extensions: Sequence[StreamerExtension] = (RescaleDownExtension(),)
    supported_kernel = SupportedKernel(kernel.RescaleOp, [i32, i8])


class RescaleUpXDMA(RescaleXDMA):
    """
    Snax XDMA Rescale Up Extension
    """

    required_extensions: Sequence[StreamerExtension] = (RescaleUpExtension(),)
    supported_kernel = SupportedKernel(kernel.RescaleOp, [i8, i32])
