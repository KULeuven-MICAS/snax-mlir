from collections.abc import Sequence

from xdsl.dialects import arith, builtin, linalg, llvm, memref
from xdsl.ir import Operation, SSAValue

from compiler.accelerators.snax import SNAXAccelerator, SNAXPollingBarrier2, SNAXStreamer
from compiler.accelerators.streamers import Streamer, StreamerConfiguration, StreamerType
from compiler.dialects import accfg, snax_stream

default_streamer = StreamerConfiguration(
    [
        Streamer(StreamerType.Reader, 6, 2),  # A
        Streamer(StreamerType.Reader, 3, 2),  # B
        Streamer(StreamerType.Writer, 3, 2),  # D8
        Streamer(StreamerType.Reader, 3, 2),  # C
        Streamer(StreamerType.Writer, 3, 2),  # D32
    ],
    separate_loop_bounds=True,
)


class SNAXGEMMXAccelerator(SNAXAccelerator, SNAXStreamer):
    """
    Accelerator Interface class for SNAX GEMMX accelerator
    """

    name = "snax_gemmx"

    def __init__(self, streamer_config: StreamerConfiguration = default_streamer) -> None:
        super().__init__(streamer_config)

        self.fields = (
            *self.streamer_setup_fields,
            "K",
            "N",
            "M",
            "subtractions",
            "csr0",
            "csr1",
            "csr2",
            "temporal_loop_bound",
            "bypassSIMD",
        )

        self.launch_fields = (*self.streamer_launch_fields, "launch_gemmx")

    def generate_acc_op(self) -> accfg.AcceleratorOp:
        """
        Return a SNAX GEMMX accelerator op
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
                "csr0": addr_next + 4,
                "csr1": addr_next + 5,
                "csr2": addr_next + 6,
                "temporal_loop_bound": addr_next + 7,
                "bypassSIMD": addr_next + 8,
            },
            {**streamer_launch, "launch_gemmx": addr_next + 9},
            addr_next + 9,
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
        c1 = arith.Constant.from_int_and_width(1, 32)
        knm = [
            (((cst := arith.Constant.from_int_and_width(val.data, 32)),), cst.result)
            for val in op.stride_patterns.data[0].upper_bounds
        ]

        streamer_setup_vals = list(self._generate_streamer_setup_vals(op))

        # FIXME:
        # override C until bias fusion is complete:
        # bounds:
        streamer_setup_vals[12] = streamer_setup_vals[15]
        streamer_setup_vals[13] = streamer_setup_vals[16]
        streamer_setup_vals[14] = streamer_setup_vals[17]
        streamer_setup_vals[15] = ([], streamer_setup_vals[15][1])
        streamer_setup_vals[16] = ([], streamer_setup_vals[16][1])
        streamer_setup_vals[17] = ([], streamer_setup_vals[17][1])
        # strides:
        streamer_setup_vals[18 + 12] = streamer_setup_vals[18 + 15]
        streamer_setup_vals[18 + 13] = streamer_setup_vals[18 + 16]
        streamer_setup_vals[18 + 14] = streamer_setup_vals[18 + 17]
        streamer_setup_vals[18 + 15] = ([], streamer_setup_vals[18 + 15][1])
        streamer_setup_vals[18 + 16] = ([], streamer_setup_vals[18 + 16][1])
        streamer_setup_vals[18 + 17] = ([], streamer_setup_vals[18 + 17][1])
        # spatial strides:
        streamer_setup_vals[36 + 6] = streamer_setup_vals[36 + 8]
        streamer_setup_vals[36 + 7] = streamer_setup_vals[36 + 9]
        streamer_setup_vals[36 + 8] = ([], streamer_setup_vals[36 + 8][1])
        streamer_setup_vals[36 + 9] = ([], streamer_setup_vals[36 + 9][1])
        # baseptr
        streamer_setup_vals[49] = streamer_setup_vals[50]
        streamer_setup_vals[50] = ([], streamer_setup_vals[50][1])



        return [
            *streamer_setup_vals,
            *knm,
            ([c0], c0.result), # subtractions
            ([], c0.result), # csr0
            ([], c0.result), # csr1
            ([], c0.result), # csr2
            ([], c0.result), # temporal_loop_bound
            ([c1], c1.result), # bypassSIMD
        ]

    @staticmethod
    def lower_acc_await(acc_op: accfg.AcceleratorOp) -> Sequence[Operation]:
        c0 = arith.Constant.from_int_and_width(0, 32)
        addr_acc = acc_op.launch_fields.data['launch_gemmx'].value.data
        addr_acc = arith.Constant.from_int_and_width(addr_acc, 32)
        addr_str = acc_op.launch_fields.data['launch_streamer'].value.data
        addr_str = arith.Constant.from_int_and_width(addr_str, 32)
        return [
            c0,
            addr_acc,
            addr_str,
            llvm.InlineAsmOp("csrw $0, $1", "I, K", [addr_str.result, c0.result], has_side_effects=True),
            llvm.InlineAsmOp("csrw $0, $1", "I, K", [addr_str.result, c0.result], has_side_effects=True),
            llvm.InlineAsmOp("csrw $0, $1", "I, K", [addr_acc.result, c0.result], has_side_effects=True),
            llvm.InlineAsmOp("csrw $0, $1", "I, K", [addr_acc.result, c0.result], has_side_effects=True),
        ]

