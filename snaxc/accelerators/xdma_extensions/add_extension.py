from collections.abc import Sequence

from xdsl.dialects.builtin import i32

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.xdma_extensions.dma_extension import DMAExtension
from snaxc.dialects import kernel


class AddExtension(DMAExtension):
    """
    Example extension for the SNAX XDMA accelerator.
    This class can be used to add custom functionality to the SNAX XDMA accelerator.
    """

    name = "add_ext"
    supported_kernel = SupportedKernel(kernel.AddOp, [i32, i32, i32])
    csr_length = 1

    def get_dma_extension_name(self) -> str:
        return self.name

    def get_dma_extension_kernel(self) -> SupportedKernel:
        return self.supported_kernel

    def get_csr_values(self, op: kernel.KernelOp) -> Sequence[int]:
        """
        Returns the CSR values for the Add extension.
        This method should be implemented to provide the specific CSR values needed for the Add extension.
        """
        # returns the number of input tensors in the add op, currently limited to 2
        return [2]
