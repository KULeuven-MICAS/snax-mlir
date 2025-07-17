import string
from abc import ABC, abstractmethod
from collections.abc import Sequence

from xdsl.dialects import arith, builtin, llvm
from xdsl.dialects.builtin import IntAttr, i32
from xdsl.dialects.scf import ConditionOp, WhileOp, YieldOp
from xdsl.ir import Operation, OpResult, SSAValue

from snaxc.accelerators.accelerator import Accelerator
from snaxc.accelerators.streamers import StreamerConfiguration
from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension
from snaxc.accelerators.streamers.extensions.transpose_extension import (
    TransposeExtension,
)
from snaxc.accelerators.streamers.streamers import (
    HasAddressRemap,
    HasBroadcast,
    HasChannelMask,
    Streamer,
    StreamerFlag,
)
from snaxc.dialects import accfg, snax_stream
from snaxc.dialects.dart import AccessPatternOp, StreamingRegionOpBase
from snaxc.dialects.snax_stream import StreamerConfigurationAttr, StreamingRegionOp
from snaxc.ir.dart.access_pattern import Template

c0_attr = builtin.IntegerAttr(0, builtin.IndexType())


class SNAXAccelerator(Accelerator, ABC):
    """
    Abstract base class for extending AcceleratorInterfaces
    with common SNAX lowerings.
    """

    def lower_acc_launch(self, launch_op: accfg.LaunchOp, acc_op: accfg.AcceleratorOp) -> Sequence[Operation]:
        field_to_csr = dict(acc_op.launch_field_items())
        ops: Sequence[Operation] = []
        for field, val in launch_op.iter_params():
            # Get the launch address
            launch_address = None
            assert "launch" in field
            launch_address = field_to_csr[field]
            assert launch_address is not None

            # Get the launch value,
            launch_value = val
            assert launch_value is not None

            ops.extend(
                [
                    addr_val := arith.ConstantOp(launch_address),
                    llvm.InlineAsmOp(
                        "csrw $0, $1",
                        # I = any 12 bit immediate, K = any 5 bit immediate
                        # The K allows LLVM to emit an `csrrwi` instruction,
                        # which has room for one 5 bit immediate only.
                        "I, K",
                        [addr_val, launch_value],
                        has_side_effects=True,
                    ),
                ]
            )

        return ops

    @staticmethod
    def lower_acc_setup(setup_op: accfg.SetupOp, acc_op: accfg.AcceleratorOp) -> Sequence[Operation]:
        field_to_csr = dict(acc_op.field_items())
        ops: Sequence[Operation] = []
        for field, val in setup_op.iter_params():
            if isinstance(val.type, builtin.IndexType):
                val_to_i32 = arith.IndexCastOp(val, builtin.i32)
                ops.append(val_to_i32)
                val = val_to_i32.result
            addr = field_to_csr[field]
            ops.extend(
                [
                    addr_val := arith.ConstantOp(addr),
                    llvm.InlineAsmOp(
                        "csrw $0, $1",
                        "I, rK",
                        [addr_val, val],
                        has_side_effects=True,
                    ),
                ]
            )
        return ops


class SNAXStreamer(ABC):
    """
    Abstract base class for SNAX Accelerators with Streamer interfaces.
    """

    streamer_config: StreamerConfigurationAttr
    streamer_names: Sequence[str]
    streamer_setup_fields: Sequence[str]
    streamer_launch_fields: Sequence[str]

    zero_address = 0x1000_0040

    def __init__(self, streamer_config: StreamerConfiguration | StreamerConfigurationAttr) -> None:
        if isinstance(streamer_config, StreamerConfiguration):
            streamer_config = StreamerConfigurationAttr(streamer_config)

        self.streamer_config = streamer_config

        # set streamer names as a, b, c, d, ...
        self.streamer_names = list(string.ascii_lowercase[: self.streamer_config.data.size()])

        if self.streamer_config.data.system_type() == StreamerSystemType.Regular:
            self.streamer_setup_fields = self.get_streamer_setup_fields()
            self.streamer_launch_fields = self.get_streamer_launch_fields()
        else:
            self.streamer_setup_fields = self.get_xdma_streamer_setup_fields()
            self.streamer_launch_fields = self.get_xdma_streamer_launch_fields()

    def _generate_streamer_setup_vals(self, op: StreamingRegionOp) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        result: Sequence[tuple[Sequence[Operation], SSAValue]] = []

        do_broadcast = [False] * len(self.streamer_config.data.streamers)

        for operand, streamer in enumerate(self.streamer_config.data.streamers):
            # streamer must generate zero pattern if the stream is coming from c0
            is_zero_pattern = False
            if isinstance(opresult := op.operands[operand], OpResult):
                is_zero_pattern = isinstance(opresult.op, arith.ConstantOp) and opresult.op.value == c0_attr

            # base pointers (low, high)
            if is_zero_pattern:
                czero = arith.ConstantOp.from_int_and_width(self.zero_address, i32)
                result.append(([czero], czero.result))
            else:
                result.append(([], op.operands[operand]))
            result.append(([c0 := arith.ConstantOp.from_int_and_width(0, i32)], c0.result))

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

            # address remap:
            if any(isinstance(opt, HasAddressRemap) for opt in streamer.opts):
                c0 = arith.ConstantOp.from_int_and_width(0, i32)
                result.append(([c0], c0.result))

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

        # transpose specifications
        for operand, streamer in enumerate(self.streamer_config.data.streamers):
            if any(isinstance(opt, TransposeExtension) for opt in streamer.opts):
                # if we want to disable transpose, we need
                # a 1 to the transpose field, as it is an
                # extension bypass signal
                c1 = arith.ConstantOp.from_int_and_width(1, i32)
                result.append(([c1], c1.result))

        for operand, streamer in enumerate(self.streamer_config.data.streamers):
            if any(isinstance(opt, HasBroadcast) for opt in streamer.opts):
                if do_broadcast[operand]:
                    c0 = arith.ConstantOp.from_int_and_width(0, i32)
                    result.append(([c0], c0.result))
                else:
                    c1 = arith.ConstantOp.from_int_and_width(1, i32)
                    result.append(([c1], c1.result))

        return result

    def get_streamer_setup_fields(self) -> Sequence[str]:
        result: list[str] = []

        for name, streamer in zip(self.streamer_names, self.streamer_config.data.streamers):
            # base pointers
            result.extend([f"{name}_ptr_low", f"{name}_ptr_high"])
            # spatial strides
            result.extend([f"{name}_sstride_{i}" for i in range(streamer.spatial_dim)])
            # temporal bounds
            result.extend([f"{name}_bound_{i}" for i in range(streamer.temporal_dim)])
            # temporal strides
            result.extend([f"{name}_tstride_{i}" for i in range(streamer.temporal_dim)])
            # options
            if any(isinstance(opt, HasAddressRemap) for opt in streamer.opts):
                result.append(f"{name}_address_remap")
            if any(isinstance(opt, HasChannelMask) for opt in streamer.opts):
                result.append(f"{name}_channel_mask")

        # transpose specifications
        for streamer, name in zip(self.streamer_config.data.streamers, self.streamer_names):
            if any(isinstance(opt, TransposeExtension) for opt in streamer.opts):
                result.append(f"{name}_transpose")

        for streamer, name in zip(self.streamer_config.data.streamers, self.streamer_names):
            if any(isinstance(opt, HasBroadcast) for opt in streamer.opts):
                result.append(f"{name}_broadcast")

        return result

    def get_streamer_launch_fields(self) -> Sequence[str]:
        return ["launch_streamer"]

    def get_xdma_streamer_setup_fields(self) -> Sequence[str]:
        """
        Get the setup fields for the xDMA streamer.
        """
        result: list[str] = []

        assert self.streamer_config.data.system_type() == StreamerSystemType.DmaExt, (
            "This method should only be called for xDMA streamer configurations"
        )

        for name, streamer in zip(self.streamer_names, self.streamer_config.data.streamers):
            # base address
            result.extend([f"{name}_ptr_low", f"{name}_ptr_high"])

        for i, (name, streamer) in enumerate(zip(self.streamer_names, self.streamer_config.data.streamers)):
            # spatial strides
            result.extend([f"{name}_sstride_{i}" for i in range(streamer.spatial_dim)])
            # temporal bounds
            result.extend([f"{name}_bound_{i}" for i in range(streamer.temporal_dim)])
            # temporal strides
            result.extend([f"{name}_tstride_{i}" for i in range(streamer.temporal_dim)])
            # options
            result.extend([f"{name}_enabled_chan"])
            if name == "b":
                result.append(f"{name}_enabled_byte")
            result.extend([f"{name}_bypass"])
            # Extensions
            for extension in streamer.opts:
                if isinstance(extension, StreamerExtension):
                    for i in range(extension.csr_length):
                        result.append(f"{name}_{extension.name}_{i}")

        return result

    def get_xdma_streamer_launch_fields(self) -> Sequence[str]:
        """
        Get the launch fields for the xDMA streamer.
        """
        return [
            "launch_start",
        ]

    def get_streamer_setup_dict(self, base_addr: int) -> tuple[int, dict[str, int]]:
        """
        Generate CSR Addresses for the setup of the streamers

        Parameters:
        base_addr (int): the base CSR address

        Returns:
        int: The next usable CSR address
        dict[str, int]: The dictionary mapping setup field to csr address
        """
        streamer_setup = {key: base_addr + i for i, key in enumerate(self.streamer_setup_fields)}
        base_addr += len(self.streamer_setup_fields)
        return base_addr, streamer_setup

    def get_streamer_launch_dict(self, base_addr: int) -> tuple[int, dict[str, int]]:
        """
        Generate CSR Addresses for the launch of the streamers

        Parameters:
        base_addr (int): the base CSR address

        Returns:
        int: The next usable CSR address
        dict[str, int]: The dictionary mapping setup field to csr address
        """
        streamer_launch = {key: base_addr + i for i, key in enumerate(self.streamer_launch_fields)}
        base_addr += len(self.streamer_launch_fields)

        # 1 busy register + 1 performance counter after launch field
        base_addr += 2

        return base_addr, streamer_launch

    @abstractmethod
    def get_template(self, op: StreamingRegionOpBase) -> Template:
        """
        Get the template for this acelerator to schedule a given
        stream.streaming_region operation.

        Returns template, template_bounds
        """
        raise NotImplementedError()

    def get_streamers(self, op: StreamingRegionOpBase) -> Sequence[Streamer]:
        """
        Return the set of streamers used for a given op.
        """
        return self.streamer_config.data.streamers

    def set_stride_patterns(
        self,
        op: AccessPatternOp,
        snax_stride_patterns: Sequence[snax_stream.StridePattern],
    ) -> tuple[
        Sequence[SSAValue],
        Sequence[SSAValue],
        Sequence[snax_stream.StridePattern],
        Sequence[Operation],
    ]:
        """
        Allows the accelerator to customize the found stride patterns
        after scheduling and layout resolution, for a given operation.

        Returns the new inputs, outputs and strides for the snax StridePattern
        operation, along with a list of ops to add to the IR.
        """
        return (op.inputs, op.outputs, snax_stride_patterns, [])


class SNAXPollingBarrier(Accelerator, ABC):
    """
    Abstract base class for SNAX Accelerators with Polling style barrier.
    In this case, the while loop makes the CPU continuously polls
    an accelerator register to see if it is finished.

    The polling style barrier can be represented in C with:

    void snax_mac_sw_barrier() {
        // poll csr 0x3c3 until HWPE MAC accelerator is finished
        while (read_csr(0x3c3)) {
        };
        // This is necessary for the HWPE MAC accelerator to allow multiple runs
        // write 0x3c5 to clear HWPE accelerator
        // Otherwise the accelerator goes into undefined behaviour:
        // It might stall/continue indefinitely
        write_csr(0x3c5, 0);
        asm volatile("nop\n");
        asm volatile("nop\n");
        asm volatile("nop\n");
    }
    """

    @staticmethod
    def lower_acc_await(acc_op: accfg.AcceleratorOp) -> Sequence[Operation]:
        # kernels/tiled_mult/tiled.preprocfinal.mlir only works
        # when at least 4 nops are introduced, due to hardware handshake issues.
        # this is will likely not be fixed in the future.
        nops = [llvm.InlineAsmOp("nop", "", [], [], has_side_effects=True) for _ in range(4)]
        return [
            WhileOp(
                [],
                [],
                [
                    barrier := arith.ConstantOp(acc_op.barrier),
                    zero := arith.ConstantOp(builtin.IntegerAttr.from_int_and_width(0, 32)),
                    status := llvm.InlineAsmOp(
                        "csrr $0, $1",
                        # I = any 12 bit immediate
                        # =r = store result in A 32- or 64-bit
                        # general-purpose register (depending on the platform XLEN)
                        "=r, I",
                        [barrier],
                        [i32],
                        has_side_effects=True,
                    ),
                    # check if not equal to zero
                    comparison := arith.CmpiOp(status, zero, "ne"),
                    ConditionOp(comparison.results[0]),
                ],
                [
                    YieldOp(),
                ],
            ),
            addr_val := arith.ConstantOp(builtin.IntegerAttr(965, 12)),  # 0x3c5 = 965
            zero := arith.ConstantOp(builtin.IntegerAttr.from_int_and_width(0, 5)),
            llvm.InlineAsmOp(
                "csrw $0, $1",
                "I, K",
                [addr_val, zero],
                has_side_effects=True,
            ),
            # a lot of nops for random but important reasons
            *nops,
        ]


class SNAXPollingBarrier2(Accelerator, ABC):
    """
    FIXME: Adapt this to an interrupt-style barrier for the newest RTL

    Abstract base class for SNAX Accelerators with different polling style barrier.

    The polling style barrier can be represented in C with:
    void wait_batch_gemm() {
        uint32_t break_poll;

        while (1) {
            // poll the state CSR[1] to see if GEMM is still busy
            break_poll = read_csr(0x3cf);
            if ((break_poll >> 1) == 1) {
                break;
            };
        };
    """

    @staticmethod
    def lower_acc_await(acc_op: accfg.AcceleratorOp) -> Sequence[Operation]:
        return [
            WhileOp(
                [],
                [],
                [
                    barrier := arith.ConstantOp(acc_op.barrier),
                    one := arith.ConstantOp(builtin.IntegerAttr.from_int_and_width(1, 32)),
                    status := llvm.InlineAsmOp(
                        "csrr $0, $1",
                        # I = any 12 bit immediate
                        # =r = store result in A 32- or 64-bit
                        # general-purpose register (depending on the platform XLEN)
                        "=r, I",
                        [barrier],
                        [i32],
                        has_side_effects=True,
                    ),
                    shifted_status := arith.ShRUIOp(status, one),
                    # check if not equal to one
                    comparison := arith.CmpiOp(shifted_status, one, "ne"),
                    ConditionOp(comparison.results[0]),
                ],
                [
                    YieldOp(),
                ],
            ),
        ]


class SNAXPollingBarrier3(Accelerator, ABC):
    """
    FIXME: Adapt this to an interrupt-style barrier for the newest RTL

    Abstract base class for SNAX Accelerators with different polling style barrier.

    The polling style barrier can be represented in C with:
        while (read_csr(0x3cf));
    """

    @staticmethod
    def lower_acc_await(acc_op: accfg.AcceleratorOp) -> Sequence[Operation]:
        return [
            WhileOp(
                [],
                [],
                [
                    barrier := arith.ConstantOp(acc_op.barrier),
                    zero := arith.ConstantOp(builtin.IntegerAttr.from_int_and_width(0, 32)),
                    status := llvm.InlineAsmOp(
                        "csrr $0, $1",
                        # I = any 12 bit immediate
                        # =r = store result in A 32- or 64-bit
                        # general-purpose register (depending on the platform XLEN)
                        "=r, I",
                        [barrier],
                        [i32],
                        has_side_effects=True,
                    ),
                    # check if not equal to zero
                    comparison := arith.CmpiOp(status, zero, "ne"),
                    ConditionOp(comparison.results[0]),
                ],
                [
                    YieldOp(),
                ],
            ),
        ]


class SNAXPollingBarrier4(Accelerator, ABC):
    """
    Polling barrier used in accelerators with streamers.
    This tries to write a 0 to every launch field twice.
    This is represented in C with:
    write_csr(launch_streamer, 0);
    write_csr(launch_streamer, 0);
    write_csr(launch_accelerator, 0);
    write_csr(launch_accelerator, 0);
    """

    @staticmethod
    def lower_acc_await(acc_op: accfg.AcceleratorOp) -> Sequence[Operation]:
        c0 = arith.ConstantOp.from_int_and_width(0, 32)
        result: list[Operation] = [c0]

        for _, launch_addr in acc_op.launch_field_items():
            addr_op = arith.ConstantOp.from_int_and_width(launch_addr.value.data, 32)
            write_op_1 = llvm.InlineAsmOp(
                "csrw $0, $1",
                "I, K",
                [addr_op.result, c0.result],
                has_side_effects=True,
            )
            write_op_2 = llvm.InlineAsmOp(
                "csrw $0, $1",
                "I, K",
                [addr_op.result, c0.result],
                has_side_effects=True,
            )
            result.extend((addr_op, write_op_1, write_op_2))

        return result
