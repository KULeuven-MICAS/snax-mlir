from collections.abc import Sequence

from xdsl.dialects.builtin import i8, i32

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension
from snaxc.dialects import kernel


class RescaleDownExtension(StreamerExtension):
    """
    Snax XDMA Rescale Down Extension
    This extension is used to perform Rescale Down operations on the XDMA core.
    """

    name = "rescale_down_ext"
    supported_kernel = SupportedKernel(kernel.RescaleOp, [i32, i8])
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

        return [
            op.input_zp.value.data,
            int.from_bytes(
                op.multiplier.data.data[0:4], byteorder="little", signed=False
            ),
            op.output_zp.value.data,
            int.from_bytes(op.shift.data.data[0:4], byteorder="little", signed=False),
        ]
