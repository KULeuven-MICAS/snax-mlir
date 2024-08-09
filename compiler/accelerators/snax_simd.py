from collections.abc import Sequence

from xdsl.dialects import arith, builtin, linalg, llvm, memref, memref_stream
from xdsl.ir import Operation, OpResult, SSAValue

from compiler.accelerators.snax import SNAXAccelerator, SNAXPollingBarrier, SNAXStreamer
from compiler.accelerators.streamers import (
    Streamer,
    StreamerConfiguration,
    StreamerType,
)
from compiler.dialects import accfg, snax_stream
from compiler.util.kernel_type import KernelType

default_streamer = StreamerConfiguration(
    [
        Streamer(StreamerType.Reader, 2, 2),
        Streamer(StreamerType.Writer, 2, 2),
    ]
)


class SNAXSimdAccelerator(SNAXAccelerator, SNAXStreamer):
    """
    Accelerator interface class for the SNAX Alu accelerator.
    """

    name = "snax_simd"

    def __init__(self, streamer_config: StreamerConfiguration = default_streamer) -> None:
        super().__init__(streamer_config)

        self.fields = (*self.streamer_setup_fields, "csr_1", "csr_2", "csr_3", "loop_bound")
        self.launch_fields = (*self.streamer_launch_fields, "launch_simd")

    def convert_to_acc_ops(self, op: snax_stream.StreamingRegionOp) -> Sequence[Operation]:
        """
        Lowers the operation to a sequence of acc_ops.
        """

        # linalg.generic lowering is stil hardcoded, but kept until
        # lowering from linalg -> snax_stream is complete
        args = self._generate_stream_setup_vals(op)

        ops_to_insert = []
        for new_ops, _ in args:
            ops_to_insert.extend(new_ops)

        return [
            *ops_to_insert,
            setup := accfg.SetupOp([val for _, val in args], self.fields, self.name),
            launch_val := arith.Constant(builtin.IntegerAttr.from_int_and_width(1, 5)),
            token := accfg.LaunchOp([launch_val, launch_val], self.launch_fields, setup),
            accfg.AwaitOp(token),
        ]

    def _generate_stream_setup_vals(
        self, op: snax_stream.StreamingRegionOp
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:

        generic_op = op.body.block.first_op
        assert isinstance(generic_op, memref_stream.GenericOp)
        assert generic_op.body.block.last_op
        yielded_op = generic_op.body.block.last_op.operands[0]
        assert isinstance(yielded_op, OpResult)

        _, (zp_out, shift, mult) = KernelType.parse_rescale(yielded_op.op)

        assert isinstance(zp_out, OpResult)
        assert isinstance(shift, OpResult)
        assert isinstance(mult, OpResult)

        max_int_i = arith.Constant.from_int_and_width(127, 8)
        min_int_i = arith.Constant.from_int_and_width(-128, 8)
        double_round = arith.Constant.from_int_and_width(1, 1)
        ops_to_add: list[Operation] = [max_int_i, min_int_i, double_round]

        # cast all values to 32 bit (unsigned)
        max_int_i = arith.ExtUIOp(max_int_i, builtin.i32)
        min_int_i = arith.ExtUIOp(min_int_i, builtin.i32)
        double_round = arith.ExtUIOp(double_round, builtin.i32)
        shift = arith.TruncIOp(shift, builtin.i32)
        mult = arith.TruncIOp(mult, builtin.i32)
        ops_to_add.extend([max_int_i, min_int_i, double_round, shift, mult])


        # shift all values to the correct amount

        max_int_i = arith.ShLI(max_int_i, (c24 := arith.Constant.from_int_and_width(24, 32)), builtin.i32)
        shift = arith.ShLI(shift, (c16 := arith.Constant.from_int_and_width(16, 32)), builtin.i32)
        zp_out = arith.ShLI(zp_out, (c8 := arith.Constant.from_int_and_width(8, 32)), builtin.i32)
        double_round = arith.ShLI(double_round, c8, builtin.i32)

        ops_to_add.extend([c24, c16, c8, max_int_i, shift, zp_out, double_round])

        # construct csr0 value (zp in is always 0 for now)
        ops_to_add.append(csr0 := arith.Addi(max_int_i, shift))
        ops_to_add.append(csr0 := arith.Addi(csr0, zp_out))


        loop_bound = arith.Constant.from_int_and_width(
            op.stride_patterns.data[0].upper_bounds.data[0].data * op.stride_patterns.data[0].upper_bounds.data[1].data,
            32,
        )

        return [
            *self._generate_streamer_setup_vals(op),
            (ops_to_add, csr0.result),
            ([], double_round.result),
            ([], mult.result),
            ([loop_bound], loop_bound.result)
        ]

    def generate_acc_op(self) -> accfg.AcceleratorOp:
        # base address:
        base_addr = 0x3C0

        # streamer setup addresses
        addr_next, streamer_setup = self.get_streamer_setup_dict(base_addr)
        # streamer launch addresses
        addr_next, streamer_launch = self.get_streamer_launch_dict(addr_next)

        op = accfg.AcceleratorOp(
            self.name,
            {
                **streamer_setup,
                "csr_1": addr_next + 0,
                "csr_2": addr_next + 1,
                "csr_3": addr_next + 2,
                "loop_bound": addr_next + 3,
            },
            {**streamer_launch, "launch_simd": addr_next + 4},
            addr_next + 5,
        )

        # add snax streamer interface
        op.attributes["streamer_config"] = self.streamer_config

        return op

    @staticmethod
    def lower_acc_await(acc_op: accfg.AcceleratorOp) -> Sequence[Operation]:
        c0 = arith.Constant.from_int_and_width(0, 32)
        addr_acc = acc_op.launch_fields.data['launch_simd'].value.data
        addr_acc = arith.Constant.from_int_and_width(addr_acc, 32)
        addr_str = acc_op.launch_fields.data['launch_streamer'].value.data
        addr_str = arith.Constant.from_int_and_width(addr_str, 32)
        return [
            c0,
            addr_acc,
            addr_str,
            llvm.InlineAsmOp("csrw $0, $1", "I, K", [addr_acc.result, c0.result], has_side_effects=True),
            llvm.InlineAsmOp("csrw $0, $1", "I, K", [addr_str.result, c0.result], has_side_effects=True),
        ]
