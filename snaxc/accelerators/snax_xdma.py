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
        self.equal_to_zero = True
        # Supported kernels are given by all available extensions
        temp_supported_kernels = [
            ext.get_dma_extension_kernel()
            for ext in default_streamer.streamers[0].opts + default_streamer.streamers[1].opts
            if isinstance(ext, StreamerExtension)
        ]
        # Filter out None values and convert to tuple
        self.supported_kernels = tuple(kernel for kernel in temp_supported_kernels if kernel is not None)

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
                    if isinstance(str_op := op.body.block.first_op, dart.GenericOp):
                        if ext.supported_kernel is not None and ext.supported_kernel.is_same_kernel(
                            kernel_op := str_op.body.block.first_op
                        ):
                            bypass -= 2**i
                    i += 1
            cst = arith.ConstantOp.from_int_and_width(bypass, i32)
            result.append(([cst], cst.result))

            # Extensions
            for ext in streamer.opts:
                if isinstance(ext, StreamerExtension):
                    if isinstance(str_op := op.body.block.first_op, dart.GenericOp):
                        if ext.supported_kernel is not None and ext.supported_kernel.is_same_kernel(
                            kernel_op := str_op.body.block.first_op
                        ):
                            # Check for each extension what its csr values are
                            assert isinstance(kernel_op, KernelOp), "Expected a KernelOp in the GenericOp"
                            for csr_val in ext.get_csr_values(kernel_op):
                                cst = arith.ConstantOp.from_int_and_width(csr_val, i32)
                                result.append(([cst], cst.result))
                        else:
                            for i in range(ext.csr_length):
                                # If the kernel is not supported, we still need to add the CSR values
                                # but they will be set to 0.
                                cst = arith.ConstantOp.from_int_and_width(0, i32)
                                result.append(([cst], cst.result))
                    else:
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
        kernel_op = op.body.block.first_op
        assert isinstance(kernel_op, dart.GenericOp), "Expected a GenericOp in the StreamingRegionOp"
        kernel_op = kernel_op.body.block.first_op
        assert isinstance(kernel_op, KernelOp), "Expected a KernelOp in the GenericOp"
        for streamer in self.streamer_config.data.streamers:
            for ext in streamer.opts:
                if isinstance(ext, StreamerExtension):
                    if ext.supported_kernel is not None and ext.supported_kernel.is_same_kernel(kernel_op):
                        return ext.get_template(kernel_op)
        raise RuntimeError("No suitable extension found for the kernel operation in the StreamingRegionOp.")

    def get_streamers(self, op: dart.StreamingRegionOpBase) -> Sequence[Streamer]:
        kernel_op = op.body.block.first_op
        assert isinstance(kernel_op, dart.GenericOp), "Expected a GenericOp in the StreamingRegionOp"
        kernel_op = kernel_op.body.block.first_op
        assert isinstance(kernel_op, KernelOp), "Expected a KernelOp in the GenericOp"
        for streamer in self.streamer_config.data.streamers:
            for ext in streamer.opts:
                if isinstance(ext, StreamerExtension):
                    if ext.supported_kernel is not None and ext.supported_kernel.is_same_kernel(kernel_op):
                        return ext.get_streamers(streamer_config=self.streamer_config.data)
        # If no specific extension is found, return the default streamers
        raise RuntimeError("No suitable extension found for the kernel operation in the StreamingRegionOp.")

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
        kernel_op = op.body.block.first_op
        assert isinstance(kernel_op, dart.GenericOp), "Expected a GenericOp in the StreamingRegionOp"
        kernel_op = kernel_op.body.block.first_op
        assert isinstance(kernel_op, KernelOp), "Expected a KernelOp in the GenericOp"
        for streamer in self.streamer_config.data.streamers:
            for ext in streamer.opts:
                if isinstance(ext, StreamerExtension):
                    if ext.supported_kernel is not None and ext.supported_kernel.is_same_kernel(kernel_op):
                        new_in, new_out, new_snax_patterns, new_ops = ext.set_stride_patterns(
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
        raise RuntimeError("No suitable extension found for the kernel operation in the AccessPatternOp.")
