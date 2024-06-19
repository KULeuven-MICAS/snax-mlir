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
            token := accfg.LaunchOp([launch_val, launch_val], self.launch_fields, setup),
            accfg.AwaitOp(token),
        ]

    def _generate_setup_vals(self, op: linalg.Generic) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        c0 = arith.Constant.from_int_and_width(0, 32)
        c4 = arith.Constant.from_int_and_width(4, 32)
        c8 = arith.Constant.from_int_and_width(8, 32)
        c32 = arith.Constant.from_int_and_width(32, 32)
        c64 = arith.Constant.from_int_and_width(64, 32)

        ptr_a = memref.ExtractAlignedPointerAsIndexOp.get(op.inputs[0])
        ptr_b = memref.ExtractAlignedPointerAsIndexOp.get(op.inputs[1])
        ptr_o = memref.ExtractAlignedPointerAsIndexOp.get(op.outputs[0])

        ptr_a_i32 = builtin.UnrealizedConversionCastOp.get([ptr_a], [builtin.i32])
        ptr_b_i32 = builtin.UnrealizedConversionCastOp.get([ptr_b], [builtin.i32])
        ptr_o_i32 = builtin.UnrealizedConversionCastOp.get([ptr_o], [builtin.i32])

        return [
            # loop bound streamer
            ([c4], c4.result),
            # temporal strides streamers
            ([c32], c32.result),
            ([], c32.result),
            ([c64], c64.result),
            # spatial strides streamers
            ([c8], c8.result),
            ([], c8.result),
            ([], c8.result),
            # base pointers streamers
            ([ptr_a, ptr_a_i32], ptr_a_i32.results[0]),
            ([ptr_b, ptr_b_i32], ptr_b_i32.results[0]),
            ([ptr_o, ptr_o_i32], ptr_o_i32.results[0]),
            # alu mode
            ([c0], c0.result),
            # alu iterations
            ([], c4.result),
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
