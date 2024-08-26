import string
from abc import ABC
from collections.abc import Sequence

from typing_extensions import Tuple
from xdsl.dialects import arith, builtin, llvm
from xdsl.dialects.builtin import IntAttr, i32
from xdsl.dialects.scf import Condition, While, Yield
from xdsl.ir import Operation, SSAValue

from compiler.accelerators.accelerator import Accelerator
from compiler.accelerators.streamers import StreamerConfiguration
from compiler.dialects import accfg
from compiler.dialects.snax_stream import StreamerConfigurationAttr, StreamingRegionOp


class SNAXAccelerator(Accelerator, ABC):
    """
    Abstract base class for extending AcceleratorInterfaces
    with common SNAX lowerings.
    """

    @staticmethod
    def lower_acc_launch(
        launch_op: accfg.LaunchOp, acc_op: accfg.AcceleratorOp
    ) -> Sequence[Operation]:
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
                    addr_val := arith.Constant(launch_address),
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
    def lower_acc_setup(
        setup_op: accfg.SetupOp, acc_op: accfg.AcceleratorOp
    ) -> Sequence[Operation]:
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
                    addr_val := arith.Constant(addr),
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

    def __init__(
        self, streamer_config: StreamerConfiguration | StreamerConfigurationAttr
    ) -> None:
        if isinstance(streamer_config, StreamerConfiguration):
            streamer_config = StreamerConfigurationAttr(streamer_config)

        self.streamer_config = streamer_config

        # set streamer names as a, b, c, d, ...
        self.streamer_names = list(
            string.ascii_lowercase[: self.streamer_config.data.size()]
        )

        self.streamer_setup_fields = self.get_streamer_setup_fields()
        self.streamer_launch_fields = self.get_streamer_launch_fields()

    def _generate_streamer_setup_vals(
        self, op: StreamingRegionOp
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        result: Sequence[tuple[Sequence[Operation], SSAValue]] = []

        # loop bound registers
        loop_bounds: Sequence[IntAttr] = op.stride_patterns.data[0].upper_bounds.data
        result.extend(
            [
                (
                    [cst := arith.Constant.from_int_and_width(loop_bound.data, i32)],
                    cst.result,
                )
                for loop_bound in loop_bounds
            ]
        )

        # temporal strides
        for streamer in range(self.streamer_config.data.size()):
            for dim in range(self.streamer_config.data.temporal_dim()):
                cst = arith.Constant.from_int_and_width(
                    op.stride_patterns.data[streamer].temporal_strides.data[dim], i32
                )
                result.append(([cst], cst.result))

        # spatial strides
        for streamer in range(self.streamer_config.data.size()):
            for dim in range(self.streamer_config.data.spatial_dim()):
                cst = arith.Constant.from_int_and_width(
                    op.stride_patterns.data[streamer].spatial_strides.data[dim], i32
                )
                result.append(([cst], cst.result))

        # input & output base pointers
        result.extend(([], x) for x in op.inputs)
        result.extend(([], x) for x in op.outputs)

        return result

    def get_streamer_setup_fields(self) -> Sequence[str]:
        result: list[str] = []

        # loop bound registers
        result.extend(
            [f"loop_bound_{i}" for i in range(self.streamer_config.data.temporal_dim())]
        )

        # temporal strides
        result.extend(
            [
                f"{streamer}_tstride_{i}"
                for streamer in self.streamer_names
                for i in range(self.streamer_config.data.temporal_dim())
            ]
        )

        # spatial strides
        result.extend(
            [
                f"{streamer}_sstride_{i}"
                for streamer in self.streamer_names
                for i in range(self.streamer_config.data.spatial_dim())
            ]
        )

        # base pointers
        result.extend([f"{streamer}_ptr" for streamer in self.streamer_names])

        return result

    def get_streamer_launch_fields(self) -> Sequence[str]:
        return ["launch_streamer"]

    def get_streamer_setup_dict(self, base_addr) -> Tuple[int, dict[str, int]]:
        """
        Generate CSR Addresses for the setup of the streamers

        Parameters:
        base_addr (int): the base CSR address

        Returns:
        int: The next usable CSR address
        dict[str, int]: The dictionary mapping setup field to csr address
        """
        streamer_setup = {
            key: base_addr + i for i, key in enumerate(self.streamer_setup_fields)
        }
        base_addr += len(self.streamer_setup_fields)
        return base_addr, streamer_setup

    def get_streamer_launch_dict(self, base_addr) -> Tuple[int, dict[str, int]]:
        """
        Generate CSR Addresses for the launch of the streamers

        Parameters:
        base_addr (int): the base CSR address

        Returns:
        int: The next usable CSR address
        dict[str, int]: The dictionary mapping setup field to csr address
        """
        streamer_launch = {
            key: base_addr + i for i, key in enumerate(self.streamer_launch_fields)
        }
        base_addr += len(self.streamer_launch_fields)

        # 1 performance counter after launch field
        base_addr += 1

        return base_addr, streamer_launch


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
        nops = [
            llvm.InlineAsmOp("nop", "", [], [], has_side_effects=True) for _ in range(4)
        ]
        return [
            While(
                [],
                [],
                [
                    barrier := arith.Constant(acc_op.barrier),
                    zero := arith.Constant(
                        builtin.IntegerAttr.from_int_and_width(0, 32)
                    ),
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
                    comparison := arith.Cmpi(status, zero, "ne"),
                    Condition(comparison.results[0]),
                ],
                [
                    Yield(),
                ],
            ),
            addr_val := arith.Constant(builtin.IntegerAttr(965, 12)),  # 0x3c5 = 965
            zero := arith.Constant(builtin.IntegerAttr.from_int_and_width(0, 5)),
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
            While(
                [],
                [],
                [
                    barrier := arith.Constant(acc_op.barrier),
                    one := arith.Constant(
                        builtin.IntegerAttr.from_int_and_width(1, 32)
                    ),
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
                    shifted_status := arith.ShRUI(status, one),
                    # check if not equal to one
                    comparison := arith.Cmpi(shifted_status, one, "ne"),
                    Condition(comparison.results[0]),
                ],
                [
                    Yield(),
                ],
            ),
        ]
