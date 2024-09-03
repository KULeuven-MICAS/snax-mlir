from collections.abc import Sequence

from xdsl.dialects import arith, builtin, linalg, memref
from xdsl.ir import Operation, SSAValue

from compiler.accelerators.snax import (
    SNAXAccelerator,
    SNAXPollingBarrier3,
    SNAXStreamer,
)
from compiler.accelerators.streamers import (
    Streamer,
    StreamerConfiguration,
    StreamerType,
)
from compiler.dialects import accfg, snax_stream

default_streamer = StreamerConfiguration(
    [
        Streamer(StreamerType.Reader, 1, 1),
        Streamer(StreamerType.Reader, 1, 1),
        Streamer(StreamerType.Writer, 1, 1),
    ]
)


class SNAXAluAccelerator(SNAXAccelerator, SNAXPollingBarrier3, SNAXStreamer):
    """
    Accelerator interface class for the SNAX Alu accelerator.
    """

    name = "snax_alu"

    def __init__(
        self, streamer_config: StreamerConfiguration = default_streamer
    ) -> None:
        super().__init__(streamer_config)

        self.fields = (*self.streamer_setup_fields, "alu_mode", "loop_bound_alu")
        self.launch_fields = (*self.streamer_launch_fields, "launch_alu")

    def convert_to_acc_ops(self, op: Operation) -> Sequence[Operation]:
        """
        Lowers the operation to a sequence of acc_ops.
        """

        # linalg.generic lowering is stil hardcoded, but kept until
        # lowering from linalg -> snax_stream is complete
        if isinstance(op, linalg.Generic):
            args = self._generate_setup_vals(op)
        elif isinstance(op, snax_stream.StreamingRegionOp):
            args = self._generate_stream_setup_vals(op)
        else:
            return []

        ops_to_insert = []
        for new_ops, _ in args:
            ops_to_insert.extend(new_ops)

        return [
            *ops_to_insert,
            setup := accfg.SetupOp([val for _, val in args], self.fields, self.name),
            launch_val := arith.Constant(builtin.IntegerAttr.from_int_and_width(1, 5)),
            token := accfg.LaunchOp(
                [launch_val, launch_val], self.launch_fields, setup
            ),
            accfg.AwaitOp(token),
        ]

    def _generate_setup_vals(
        self, op: linalg.Generic
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        a, b, o = op.operands

        c0_index = arith.Constant.from_int_and_width(0, builtin.IndexType())
        dim_0 = memref.Dim.from_source_and_index(a, c0_index)
        design_time_parallelism = arith.Constant.from_int_and_width(
            4, builtin.IndexType()
        )
        loop_bound = arith.DivUI(dim_0, design_time_parallelism)
        loop_bound_i32 = arith.IndexCastOp(loop_bound, builtin.i32)
        c0 = arith.Constant.from_int_and_width(0, 32)
        c8 = arith.Constant.from_int_and_width(8, 32)
        c32 = arith.Constant.from_int_and_width(32, 32)

        ptrs = [
            (
                [
                    ptr := memref.ExtractAlignedPointerAsIndexOp.get(ref),
                    metadata := memref.ExtractStridedMetaDataOp(ref),
                    el_bytes := arith.Constant.from_int_and_width(
                        ref.type.element_type.size, builtin.IndexType()
                    ),
                    byte_offset := arith.Muli(metadata.offset, el_bytes),
                    ptr_plus_byte_offset := arith.Addi(
                        ptr, byte_offset, builtin.IndexType()
                    ),
                    ptr_i32 := arith.IndexCastOp(ptr_plus_byte_offset, builtin.i32),
                ],
                ptr_i32.result,
            )
            for ref in (a, b, o)
        ]

        return [
            # loop bound streamer
            (
                [c0_index, dim_0, design_time_parallelism, loop_bound, loop_bound_i32],
                loop_bound_i32.result,
            ),
            # temporal strides streamers
            ([c32], c32.result),
            ([], c32.result),
            ([], c32.result),
            # spatial strides streamers
            ([c8], c8.result),
            ([], c8.result),
            ([], c8.result),
            # base pointers streamers
            (ptrs[0]),
            (ptrs[1]),
            (ptrs[2]),
            # alu mode
            ([c0], c0.result),
            # alu iterations
            ([], loop_bound_i32.result),
        ]

    def _generate_stream_setup_vals(
        self, op: snax_stream.StreamingRegionOp
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        c0 = arith.Constant.from_int_and_width(0, 32)
        loop_bound = arith.Constant.from_int_and_width(
            op.stride_patterns.data[0].upper_bounds.data[0], 32
        )

        return [
            *self._generate_streamer_setup_vals(op),
            ([c0], c0.result),
            ([loop_bound], loop_bound.result),
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
                "alu_mode": addr_next + 0,
                "loop_bound_alu": addr_next + 1,
            },
            {**streamer_launch, "launch_alu": addr_next + 2},
            addr_next + 3,
        )

        # add snax streamer interface
        op.attributes["streamer_config"] = self.streamer_config

        return op
