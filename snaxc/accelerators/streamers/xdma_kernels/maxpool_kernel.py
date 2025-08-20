from collections.abc import Sequence

from xdsl.dialects.builtin import ArrayAttr, IntAttr, i8
from xdsl.ir import Operation, ParametrizedAttribute, SSAValue
from xdsl.ir.affine import AffineMap

from snaxc.accelerators.dispatching import SupportedKernel
from snaxc.accelerators.streamers.extensions.maxpool_extension import MaxPoolExtension
from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension
from snaxc.accelerators.streamers.streamers import (
    Streamer,
    StreamerConfiguration,
    StreamerType,
)
from snaxc.accelerators.streamers.xdma_kernels.xdma_kernel import XDMAKernel
from snaxc.dialects import dart, kernel, snax_stream
from snaxc.ir.dart.access_pattern import Template, TemplatePattern


class MaxPoolXDMA(XDMAKernel):
    """
    Snax XDMA MaxPool Extension
    This extension is used to perform max pooling operations on the XDMA core.
    """

    required_extensions: Sequence[StreamerExtension] = (MaxPoolExtension(),)
    supported_kernel = SupportedKernel(kernel.MaxPoolOp, [i8, i8, i8])

    def get_csr_values(self, op: kernel.KernelOp) -> Sequence[Sequence[int]]:
        """
        Returns the CSR values for the MaxPool extension.
        This method should be implemented to provide the specific CSR values needed for the MaxPool extension.
        """
        # Example implementation, replace with actual logic
        assert op.parent is not None, "Operation must be in a StreamingRegionOp"
        assert op.parent.parent is not None, "Operation's parent must be in a StreamingRegionOp"
        assert op.parent.parent.parent is not None, "Operation's must be in a StreamingRegionOp"
        assert op.parent.parent.parent.parent is not None, "Operation's must be in a StreamingRegionOp"
        assert op.parent.parent.parent.parent.parent is not None, "Operation must be in StreamingRegionOp"
        assert isinstance(
            streaming_region_op := op.parent.parent.parent.parent.parent.parent,
            snax_stream.StreamingRegionOp,
        ), "Operation must be in AccessPatternOp"

        assert streaming_region_op.properties["stride_patterns"] is not None, (
            "StreamingRegionOp must have stride pattern property"
        )
        assert isinstance(
            stride_pattern := streaming_region_op.properties["stride_patterns"],
            ArrayAttr,
        ), "Stride patterns must be an Array"
        assert isinstance(
            stride_pattern.data,  # pyright: ignore[reportUnknownMemberType]
            tuple,
        ), "Stride patterns must be a tuple"
        stride_pattern = stride_pattern.data  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        assert (
            len(
                stride_pattern  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            )
            == 2
        ), "Stride patterns tuple must have length 2"
        assert isinstance(
            stride_pattern[1],  # pyright: ignore[reportUnknownMemberType]
            snax_stream.StridePattern,
        ), "Stride patterns must contain a StridePattern for the output"

        stride_pattern = stride_pattern[1]  # pyright: ignore[reportUnknownMemberType]

        assert isinstance(stride_pattern, snax_stream.StridePattern), "Stride pattern must be a StridePattern"

        assert isinstance(stride_array := stride_pattern.parameters[0], ArrayAttr), (
            "Stride pattern parameters must be ParametrizedAttributes"
        )

        assert len(stride_array.data) >= 1, "Stride pattern must have one parameter"  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        assert isinstance(kernel_size := stride_array.data[0], IntAttr)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        assert isinstance(kernel_size := kernel_size.data, int), "Stride pattern first parameter must be an IntAttr"

        return [[kernel_size]]

    def get_template(self, op: kernel.KernelOp) -> Template:
        template = [
            AffineMap.from_callable(lambda y: (y,)),
            AffineMap.from_callable(lambda _: tuple()),
            AffineMap.from_callable(lambda y: (y,)),
        ]
        template_bounds = (64,)
        return Template(TemplatePattern(template_bounds, tp) for tp in template)

    def get_streamers(self, streamer_config: StreamerConfiguration) -> Sequence[Streamer]:
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
