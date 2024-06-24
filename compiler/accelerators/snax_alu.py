from collections.abc import Sequence

from xdsl.dialects import arith, builtin, linalg, memref
from xdsl.ir import Operation, SSAValue

from compiler.accelerators.snax import SNAXAccelerator, SNAXPollingBarrier
from compiler.dialects import accfg


class SNAXAluAccelerator(SNAXAccelerator, SNAXPollingBarrier):
    """
    Accelerator interface class for the SNAX Alu accelerator.
    """

    name = "snax_alu"
    fields = (
        "loop_bound_streamer",
        "a_tstride",
        "b_tstride",
        "o_tstride",
        "a_sstride",
        "b_sstride",
        "o_sstride",
        "a_ptr",
        "b_ptr",
        "o_ptr",
        "alu_mode",
        "loop_bound_alu",
    )
    launch_fields = ("launch_streamer", "launch_alu")

    def convert_to_acc_ops(self, op: linalg.Generic) -> Sequence[Operation]:
        """
        Lowers the operation to a sequence of acc_ops.
        """

        args = self._generate_setup_vals(op)

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
                        ref.type.element_type.width.data // 8, builtin.IndexType()
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

    def generate_acc_op(self) -> accfg.AcceleratorOp:
        return accfg.AcceleratorOp(
            self.name,
            {
                "loop_bound_streamer": 0x3C0,
                "a_tstride": 0x3C1,
                "b_tstride": 0x3C2,
                "o_tstride": 0x3C3,
                "a_sstride": 0x3C4,
                "b_sstride": 0x3C5,
                "o_sstride": 0x3C6,
                "a_ptr": 0x3C7,
                "b_ptr": 0x3C8,
                "o_ptr": 0x3C9,
                "alu_mode": 0x3CC,
                "loop_bound_alu": 0x3CD,
            },
            {"launch_streamer": 0x3CA, "launch_alu": 0x3CE},
            0x3CF,
        )
