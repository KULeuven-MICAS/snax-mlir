from abc import ABC
from collections.abc import Sequence

from xdsl.dialects.builtin import i8, i32

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension
from snaxc.dialects import kernel


class RescaleExtension(StreamerExtension, ABC):
    """
    Snax XDMA Rescale Extension
    This extension is used to perform Rescale operations on the XDMA core.
    """

    name: str
    supported_kernel: SupportedKernel | None
    csr_length = 4

    def get_dma_extension_name(self) -> str:
        return self.name

    def get_dma_extension_kernel(self) -> SupportedKernel | None:
        return self.supported_kernel

    def get_csr_values(self, op: kernel.KernelOp) -> Sequence[int]:
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
            op.input_zp.value.data,
            multiplier,
            op.output_zp.value.data,
            shift,
        ]


class RescaleDownExtension(RescaleExtension):
    """
    Snax XDMA Rescale Down Extension
    """

    supported_kernel = SupportedKernel(kernel.RescaleOp, [i32, i8])
    name = "rescale_down_ext"


class RescaleUpExtension(RescaleExtension):
    """
    Snax XDMA Rescale Up Extension
    """

    supported_kernel = SupportedKernel(kernel.RescaleOp, [i8, i32])
    name = "rescale_up_ext"
