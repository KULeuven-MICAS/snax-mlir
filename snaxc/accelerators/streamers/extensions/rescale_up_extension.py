from collections.abc import Sequence

from xdsl.dialects.builtin import i8, i32

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension
from snaxc.dialects import kernel


class RescaleUpExtension(StreamerExtension):
    """
    Snax XDMA Rescale Up Extension
    This extension is used to perform Rescale Up operations on the XDMA core.
    """

    name = "rescale_up_ext"
    supported_kernel = SupportedKernel(kernel.RescaleOp, [i8, i32])
    csr_length = 4

    def get_dma_extension_name(self) -> str:
        return self.name

    def get_dma_extension_kernel(self) -> SupportedKernel | None:
        return self.supported_kernel

    def get_csr_values(self, op: kernel.KernelOp) -> Sequence[int]:
        """
        Returns the CSR values for the Rescale Up extension.
        This method should be implemented to provide the specific CSR values needed for the Rescale Up extension.
        """
        # returns the number of input tensors in the rescale up op, currently limited to 2
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
