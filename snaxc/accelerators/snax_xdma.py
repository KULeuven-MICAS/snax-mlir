from collections.abc import Sequence

from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import IntAttr, i32
from xdsl.ir import Operation, OpResult, SSAValue

from snaxc.accelerators.configurable_accelerator import ConfigurableAccelerator
from snaxc.accelerators.dispatching import DispatchTemplate
from snaxc.accelerators.snax import (
    SNAXAccelerator,
    SNAXPollingBarrier5,
    SNAXStreamer,
)
from snaxc.accelerators.streamers import (
    HasBroadcast,
    HasByteMask,
    HasChannelMask,
)
from snaxc.accelerators.streamers.extensions import (
    AddExtension,
    MaxPoolExtension,
    MemSetExtension,
    RescaleDownExtension,
    RescaleUpExtension,
    StreamerExtension,
    TransposeExtension,
)
from snaxc.accelerators.streamers.streamers import (
    Streamer,
    StreamerConfiguration,
    StreamerFlag,
    StreamerSystemType,
    StreamerType,
)
from snaxc.accelerators.streamers.xdma_kernels import XDMA_KERNEL_SET
from snaxc.accelerators.streamers.xdma_kernels.xdma_kernel import XDMAKernel
from snaxc.dialects import accfg, dart, snax_stream
from snaxc.dialects.kernel import KernelOp

default_streamer = StreamerConfiguration(
    [
        Streamer(
            StreamerType.Reader,
            ["n", "n", "n", "n", "n"],
            [8],
            [
                MaxPoolExtension(),
                AddExtension(),
                RescaleDownExtension(),
                RescaleUpExtension(),
                HasChannelMask(),
            ],
        ),
        Streamer(
            StreamerType.Writer,
            ["r", "n", "n", "n", "n"],
            [8],
            [MemSetExtension(), TransposeExtension(), HasChannelMask(), HasByteMask()],
        ),
    ],
    StreamerSystemType.DmaExt,
)

c0_attr = builtin.IntegerAttr(0, builtin.IndexType())


class SNAXXDMAAccelerator(
    SNAXAccelerator,
    SNAXPollingBarrier5,
    SNAXStreamer,
    DispatchTemplate,
    ConfigurableAccelerator,
):
    """
    Accelerator interface class for the SNAX XDMA.
    """

    name = "snax_xdma"

    supported_kernels = ()
    max_multicast_dest = 25

    def __init__(self, streamer_config: StreamerConfiguration = default_streamer) -> None:
        assert default_streamer.size() == 2, "SNAX XDMA only supports two streamers (reader and writer)."

        super().__init__(streamer_config)

        self.fields = self.streamer_setup_fields
        self.launch_fields = self.streamer_launch_fields

        # Supported kernels are given by all available extensions
        self.supported_kernels = [
            xdma_kernel.supported_kernel
            for xdma_kernel in XDMA_KERNEL_SET
            if all(
                extension in default_streamer.streamers[0].opts or extension in default_streamer.streamers[1].opts
                for extension in xdma_kernel.required_extensions
            )
        ]

    def convert_to_acc_ops(self, op: Operation) -> Sequence[Operation]:
        """
        Lowers the operation to a sequence of acc_ops.
        """

        # Streamer bounds, strides and base addresses
        if isinstance(op, snax_stream.StreamingRegionOp):
            setup_args = self._generate_stream_setup_vals(op)
            launch_args = self._generate_streamer_launch_vals()
        else:
            return []

        ops_to_insert: Sequence[Operation] = []
        for new_ops, _ in setup_args:
            ops_to_insert.extend(new_ops)
        for new_ops, _ in launch_args:
            ops_to_insert.extend(new_ops)

        return [
            *ops_to_insert,
            setup := accfg.SetupOp([val for _, val in setup_args], self.fields, self.name),
            token := accfg.LaunchOp([val for _, val in launch_args], self.launch_fields, setup),
            accfg.AwaitOp(token),
        ]

    def _generate_stream_setup_vals(
        self, op: snax_stream.StreamingRegionOp
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        result: Sequence[tuple[Sequence[Operation], SSAValue]] = []

        do_broadcast = [False] * len(self.streamer_config.data.streamers)
        is_zero_pattern = False

        for operand, streamer in enumerate(self.streamer_config.data.streamers):
            # streamer must generate zero pattern if the stream is coming from c0
            is_zero_pattern = False
            if isinstance(opresult := op.operands[operand], OpResult):
                is_zero_pattern = (
                    isinstance(opresult.op, arith.ConstantOp)
                    and opresult.op.value == c0_attr  # TODO: check what zero patterns are and if they are relevant here
                )

            # base pointers (low, high)
            if is_zero_pattern:
                czero = arith.ConstantOp.from_int_and_width(self.zero_address, i32)
                result.append(([czero], czero.result))
            else:
                result.append(([], op.operands[operand]))
            result.append(([c0 := arith.ConstantOp.from_int_and_width(0, i32)], c0.result))

        # Find kernel operation and check if it is supported
        kernel_op = op.body.block.first_op
        assert isinstance(kernel_op, dart.GenericOp), "Expected a GenericOp in the StreamingRegionOp"
        kernel_op = kernel_op.body.block.first_op
        assert isinstance(kernel_op, KernelOp), "Expected a KernelOp in the GenericOp"

        used_kernel = None
        required_extensions = None
        for xdma_kernel in XDMA_KERNEL_SET:
            if xdma_kernel.supported_kernel.is_same_kernel(kernel_op):
                required_extensions = xdma_kernel.required_extensions
                used_kernel = xdma_kernel
                break

        if required_extensions is None:
            raise RuntimeError("No suitable XDMA kernel found for the operation in the StreamingRegionOp.")

        if used_kernel is None:
            raise RuntimeError("No suitable XDMA kernel found for the operation in the StreamingRegionOp.")
        assert issubclass(used_kernel, XDMAKernel), (
            "No suitable XDMA kernel found for the operation in the StreamingRegionOp."
        )
        current_required_extension_index_bypass = 0
        current_required_extension_index_csr = 0

        # Generate csr values further
        for operand, streamer in enumerate(self.streamer_config.data.streamers):
            # spatial strides
            for dim, flag in enumerate(streamer.spatial_dims):
                stride = op.stride_patterns.data[operand].spatial_strides.data[dim].data
                if stride == 0 and any(isinstance(opt, HasBroadcast) for opt in streamer.opts):
                    do_broadcast[operand] = True
                cst = arith.ConstantOp.from_int_and_width(stride, i32)
                result.append(([cst], cst.result))

            # loop bounds
            upper_bounds = op.stride_patterns.data[operand].upper_bounds.data
            # pad unused temporal bounds with 1's'
            upper_bounds = upper_bounds + ((IntAttr(1),) * (streamer.temporal_dim - len(upper_bounds)))

            # temporal strides
            temporal_strides = op.stride_patterns.data[operand].temporal_strides.data
            # pad unused spatial strides with 0's
            temporal_strides = temporal_strides + ((IntAttr(0),) * (streamer.temporal_dim - len(temporal_strides)))

            # ops for loop bounds
            for dim, flag in enumerate(streamer.temporal_dims):
                bound = upper_bounds[dim].data
                stride = temporal_strides[dim].data
                if flag == StreamerFlag.Reuse and bound > 1 and stride == 0:
                    # if internal reuse, bound can be set to 1
                    bound = 1
                cst = arith.ConstantOp.from_int_and_width(bound, i32)
                result.append(([cst], cst.result))

            # ops for temporal strides
            for dim, flag in enumerate(streamer.temporal_dims):
                stride = temporal_strides[dim].data
                if flag == StreamerFlag.Irrelevant:
                    # Irrelevant temporal strides should be zero
                    assert stride == 0
                cst = arith.ConstantOp.from_int_and_width(stride, i32)
                result.append(([cst], cst.result))

            # channel mask option
            if any(isinstance(opt, HasChannelMask) for opt in streamer.opts):
                if is_zero_pattern:
                    # mask all channels such that they generate zeros
                    c0 = arith.ConstantOp.from_int_and_width(0, i32)
                    result.append(([c0], c0.result))
                else:
                    # else, set to 32b111...111 (=-1) (all enabled)
                    n1 = arith.ConstantOp.from_int_and_width(-1, i32)
                    result.append(([n1], n1.result))

            # byte mask option
            if any(isinstance(opt, HasByteMask) for opt in streamer.opts):
                if is_zero_pattern:
                    # mask all bytes such that they generate zeros
                    c0 = arith.ConstantOp.from_int_and_width(0, i32)
                    result.append(([c0], c0.result))
                else:
                    # else, set to 32b111...111 (=-1) (all enabled)
                    n1 = arith.ConstantOp.from_int_and_width(-1, i32)
                    result.append(([n1], n1.result))

            # Bypass option
            bypass = 2 ** len([opt for opt in streamer.opts if isinstance(opt, StreamerExtension)]) - 1
            i = 0
            for ext in streamer.opts:
                if isinstance(ext, StreamerExtension):
                    if (
                        current_required_extension_index_bypass < len(required_extensions)
                        and required_extensions[current_required_extension_index_bypass] == ext
                    ):
                        current_required_extension_index_bypass += 1
                        bypass -= 2**i
                    i += 1
            cst = arith.ConstantOp.from_int_and_width(bypass, i32)
            result.append(([cst], cst.result))

            # Extensions
            for ext in streamer.opts:
                if isinstance(ext, StreamerExtension):
                    if (
                        current_required_extension_index_csr < len(required_extensions)
                        and required_extensions[current_required_extension_index_csr] == ext
                    ):
                        for csr_val in used_kernel().get_csr_values(kernel_op)[current_required_extension_index_csr]:
                            cst = arith.ConstantOp.from_int_and_width(csr_val, i32)
                            result.append(([cst], cst.result))
                            current_required_extension_index_csr += 1
                    else:
                        for i in range(ext.csr_length):
                            # If the kernel is not supported, we still need to add the CSR values
                            # but they will be set to 0.
                            cst = arith.ConstantOp.from_int_and_width(0, i32)
                            result.append(([cst], cst.result))

        return result

    def _generate_streamer_launch_vals(
        self,
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        """
        Generates the values for the streamer launch operation.
        """
        result: Sequence[tuple[Sequence[Operation], SSAValue]] = []

        # Start
        cst = arith.ConstantOp.from_int_and_width(1, i32)
        result.append(([cst], cst.result))

        return result

    def generate_acc_op(self) -> accfg.AcceleratorOp:
        # base address:
        base_addr = 0x3C0

        # streamer setup addresses
        addr_next, streamer_setup = self.get_xdma_streamer_setup_dict(base_addr)
        # streamer launch addresses
        addr_next, streamer_launch = self.get_xdma_streamer_launch_dict(addr_next)

        op = accfg.AcceleratorOp(
            self.name,
            {
                **streamer_setup,
            },
            {**streamer_launch},
            addr_next + 2,
        )

        # add snax streamer interface
        op.attributes["streamer_config"] = self.streamer_config

        return op

    def get_xdma_streamer_setup_dict(self, base_addr: int = 0x3C0) -> tuple[int, dict[str, int]]:
        streamer_setup = {key: base_addr + i for i, key in enumerate(self.streamer_setup_fields[0:4])}
        streamer_setup.update(
            {
                key: base_addr + i + 2 + 2 * self.max_multicast_dest
                for i, key in enumerate(self.streamer_setup_fields[4:])
            }
        )
        updated_base_addr = base_addr + len(self.streamer_setup_fields) + 2 * self.max_multicast_dest - 2
        return updated_base_addr, streamer_setup

    def get_xdma_streamer_launch_dict(self, base_addr: int = 0x3C0) -> tuple[int, dict[str, int]]:
        streamer_launch = {key: base_addr + i for i, key in enumerate(self.streamer_launch_fields)}
        updated_base_addr = base_addr + len(self.streamer_launch_fields)
        return (
            updated_base_addr,
            streamer_launch,
        )

    def get_template(self, op: dart.StreamingRegionOpBase):
        # Find kernel operation and check if it is supported
        kernel_op = op.body.block.first_op
        assert isinstance(kernel_op, dart.GenericOp), "Expected a GenericOp in the StreamingRegionOp"
        kernel_op = kernel_op.body.block.first_op
        assert isinstance(kernel_op, KernelOp), "Expected a KernelOp in the GenericOp"

        used_kernel = None
        for xdma_kernel in XDMA_KERNEL_SET:
            if xdma_kernel.supported_kernel.is_same_kernel(kernel_op):
                used_kernel = xdma_kernel
                break

        if used_kernel is None:
            raise RuntimeError("No suitable XDMA kernel found for the operation in the StreamingRegionOp.")
        assert issubclass(used_kernel, XDMAKernel), (
            "No suitable XDMA kernel found for the operation in the StreamingRegionOp."
        )

        return used_kernel().get_template(kernel_op)

    def get_streamers(self, op: dart.StreamingRegionOpBase) -> Sequence[Streamer]:
        # Find kernel operation and check if it is supported
        kernel_op = op.body.block.first_op
        assert isinstance(kernel_op, dart.GenericOp), "Expected a GenericOp in the StreamingRegionOp"
        kernel_op = kernel_op.body.block.first_op
        assert isinstance(kernel_op, KernelOp), "Expected a KernelOp in the GenericOp"

        used_kernel = None
        required_extensions = None
        for xdma_kernel in XDMA_KERNEL_SET:
            if xdma_kernel.supported_kernel.is_same_kernel(kernel_op):
                required_extensions = xdma_kernel.required_extensions
                used_kernel = xdma_kernel
                break

        if required_extensions is None:
            raise RuntimeError("No suitable XDMA kernel found for the operation in the StreamingRegionOp.")

        if used_kernel is None:
            raise RuntimeError("No suitable XDMA kernel found for the operation in the StreamingRegionOp.")
        assert issubclass(used_kernel, XDMAKernel), (
            "No suitable XDMA kernel found for the operation in the StreamingRegionOp."
        )
        return used_kernel().get_streamers(self.streamer_config.data)

    def set_stride_patterns(
        self,
        op: dart.AccessPatternOp,
        snax_stride_patterns: Sequence[snax_stream.StridePattern],
    ) -> tuple[
        Sequence[SSAValue],
        Sequence[SSAValue],
        Sequence[snax_stream.StridePattern],
        Sequence[Operation],
    ]:
        # Find kernel operation and check if it is supported
        kernel_op = op.body.block.first_op
        assert isinstance(kernel_op, dart.GenericOp), "Expected a GenericOp in the StreamingRegionOp"
        kernel_op = kernel_op.body.block.first_op
        assert isinstance(kernel_op, KernelOp), "Expected a KernelOp in the GenericOp"

        used_kernel = None
        for xdma_kernel in XDMA_KERNEL_SET:
            if xdma_kernel.supported_kernel.is_same_kernel(kernel_op):
                used_kernel = xdma_kernel
                break

        if used_kernel is None:
            raise RuntimeError("No suitable XDMA kernel found for the operation in the StreamingRegionOp.")
        assert issubclass(used_kernel, XDMAKernel), (
            "No suitable XDMA kernel found for the operation in the StreamingRegionOp."
        )
        new_in, new_out, new_snax_patterns, new_ops = used_kernel().set_stride_patterns(
            op, kernel_op, snax_stride_patterns
        )
        # Ensure new_snax_patterns is of type Sequence[snax_stream.StridePattern]
        new_snax_patterns_casted = [
            pattern for pattern in new_snax_patterns if isinstance(pattern, snax_stream.StridePattern)
        ]
        return (
            new_in,
            new_out,
            new_snax_patterns_casted,
            new_ops,
        )
