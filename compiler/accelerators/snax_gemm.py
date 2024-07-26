from collections.abc import Sequence

from xdsl.dialects import arith, builtin, linalg, llvm, memref
from xdsl.dialects.scf import Condition, While
from xdsl.ir import Operation, SSAValue

from compiler.accelerators.snax import SNAXAccelerator, SNAXPollingBarrier, SNAXStreamer
from compiler.accelerators.streamers import Streamer, StreamerConfiguration, StreamerType
from compiler.dialects import accfg, snax_stream

default_streamer = StreamerConfiguration(
    [Streamer(StreamerType.Reader, 3, 2), Streamer(StreamerType.Reader, 3, 2), Streamer(StreamerType.Writer, 3, 2)]
)


class SNAXGEMMAccelerator(SNAXAccelerator, SNAXStreamer):
    """
    Accelerator Interface class for SNAX GEMM accelerator
    CSR lowerings are inherited from SNAXAcceleratorInterface.

    """

    name = "snax_gemm"

    def __init__(self, streamer_config: StreamerConfiguration = default_streamer) -> None:
        super().__init__(streamer_config)

        self.fields = (*self.streamer_setup_fields, "K", "N", "M", "subtractions")

        self.launch_fields = (*self.streamer_launch_fields, "launch_gemm")

    def generate_acc_op(self) -> accfg.AcceleratorOp:
        """
        Return a SNAX GEMM accelerator op with some default field adresses
        """

        # base address:
        base_addr = 0x3C0

        addr_next, streamer_setup = self.get_streamer_setup_dict(base_addr)
        addr_next, streamer_launch = self.get_streamer_launch_dict(addr_next)

        op = accfg.AcceleratorOp(
            self.name,
            {
                **streamer_setup,
                "K": addr_next + 0,
                "N": addr_next + 1,
                "M": addr_next + 2,
                "subtractions": addr_next + 3,
            },
            {**streamer_launch, "launch_gemm": addr_next + 4},
            #TODO: find out why the barrier value does not increase
            addr_next + 4,
        )
        op.attributes["streamer_config"] = self.streamer_config
        return op

    def convert_to_acc_ops(self, op: snax_stream.StreamingRegionOp) -> Sequence[Operation]:
        """
        Lowers the operation op to a sequence of acc_ops.
        acc_ops are:
            - *.op that generates SSAValues consumed by acc2.setup
            - acc2.setup
            - acc2.launch
            - acc2.await
        These ops can further be lowered by specific instances of the
        Accelerator interface
        """
        args = self._generate_setup_vals(op)

        ops_to_insert = []
        # insert ops to calculate arguments
        for new_ops, _ in args:
            ops_to_insert.extend(new_ops)

        return [
            *ops_to_insert,
            setup := accfg.SetupOp([val for _, val in args], self.fields, self.name),
            launch_val := arith.Constant(builtin.IntegerAttr.from_int_and_width(1, 5)),
            token := accfg.LaunchOp([launch_val, launch_val], self.launch_fields, setup),
            accfg.AwaitOp(token),
        ]

    def _generate_setup_vals(self, op: snax_stream.StreamingRegionOp) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        """
        Produce a `Sequence[Operation], SSAValue` tuple
        for each field that contains:

        - a list of operations that calculate the field value
        - a reference to the SSAValue containing the calculated field value
        """

        c0 = arith.Constant.from_int_and_width(0, 32)
        knm = [
            (((cst := arith.Constant.from_int_and_width(val.data, 32)),), cst.result)
            for val in op.stride_patterns.data[0].upper_bounds
        ]

        return [*self._generate_streamer_setup_vals(op), *knm, ([c0], c0.result)]

    @staticmethod
    def lower_acc_await(acc_op: accfg.AcceleratorOp) -> Sequence[Operation]:
        c0 = arith.Constant.from_int_and_width(0, 32)
        addr_acc = acc_op.launch_fields.data['launch_gemm'].value.data
        addr_acc = arith.Constant.from_int_and_width(addr_acc, 32)
        addr_str = acc_op.launch_fields.data['launch_streamer'].value.data
        addr_str = arith.Constant.from_int_and_width(addr_str, 32)
        return [
            c0,
            addr_acc,
            addr_str,
            llvm.InlineAsmOp("csrw $0, $1", "I, K", [addr_acc.result, c0.result], has_side_effects=True),
            llvm.InlineAsmOp("csrw $0, $1", "I, K", [addr_acc.result, c0.result], has_side_effects=True),
            llvm.InlineAsmOp("csrw $0, $1", "I, K", [addr_str.result, c0.result], has_side_effects=True),
        ]
