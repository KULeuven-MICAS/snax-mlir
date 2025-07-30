from collections.abc import Sequence

from xdsl.dialects.builtin import i8, i32
from xdsl.ir.affine import AffineMap

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension
from snaxc.accelerators.streamers.streamers import Streamer, StreamerConfiguration
from snaxc.dialects import kernel
from snaxc.ir.dart.access_pattern import Template, TemplatePattern


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
            int.from_bytes(op.multiplier.data.data[0:4], byteorder="little", signed=False),
            op.output_zp.value.data,
            int.from_bytes(op.shift.data.data[0:4], byteorder="little", signed=False),
        ]

    def get_template(self, op: kernel.KernelOp) -> Template:
        """
        Returns the template for this DMA extension.
        """
        template = [AffineMap.from_callable(lambda y: (y,))] * 2  # TODO: check if this is correct
        template_bounds = (16,)
        return Template(TemplatePattern(template_bounds, tp) for tp in template)

    def get_streamers(self, streamer_config: StreamerConfiguration) -> Sequence[Streamer]:
        """
        Returns the streamers for this DMA extension.
        This method should be implemented to provide the specific streamers needed for the Add extension.
        """
        return [
            streamer_config.streamers[0],
            streamer_config.streamers[1],
        ]
