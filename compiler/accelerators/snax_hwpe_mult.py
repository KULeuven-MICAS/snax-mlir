from collections.abc import Sequence

from xdsl.dialects import arith, builtin, linalg, llvm, memref
from xdsl.dialects.builtin import i32
from xdsl.dialects.scf import Condition, While, Yield
from xdsl.ir import Operation, SSAValue

from compiler.accelerators.accelerator import AcceleratorInfo
from compiler.dialects import acc


class HWPEAcceleratorInfo(AcceleratorInfo):
    name = "snax_hwpe_mult"

    fields = ("A", "B", "O", "vector_length", "nr_iters", "mode")

    def generate_setup_vals(
        self, op: linalg.Generic
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        """
        Produce a `Sequence[Operation], SSAValue` tuple for each field that contains:

        - a list of operations that calculate the field value
        - a reference to the SSAValue containing the calculated field value
        """
        a, b, c = op.operands

        zero = arith.Constant.from_int_and_width(0, builtin.IndexType())
        iters_one = arith.Constant.from_int_and_width(1, 32)
        mode_one = arith.Constant.from_int_and_width(1, 32)
        dim = memref.Dim.from_source_and_index(a, zero)
        dim_i32 = arith.IndexCastOp(dim, builtin.i32)
        vector_length = [zero, dim, dim_i32], dim_i32.result

        nr_iters = [iters_one], iters_one.result
        mode = [mode_one], mode_one.result

        ptrs = [
            (
                [
                    ptr := memref.ExtractAlignedPointerAsIndexOp.get(ref),
                    ptr_i32 := arith.IndexCastOp(ptr, builtin.i32),
                ],
                ptr_i32.result,
            )
            for ref in (a, b, c)
        ]

        return ptrs + [nr_iters] + [vector_length] + [mode]

    def generate_acc_op(self) -> acc.AcceleratorOp:
        """
        Return this accelerator op:

        "acc2.accelerator"() <{
            name            = @snax_hwpe_mult,
            fields          = {A=0x3d0, B=0x3d1, O=0x3d3, n_iters=0x3d4,
                               vector_length=0x3d5, mode=0x3d6},
            launch_addr     = 0x3c0,
            barrier = 0x3c3,
        }> : () -> ()
        """
        return acc.AcceleratorOp(
            self.name,
            {
                "A": 0x3D0,
                "B": 0x3D1,
                "O": 0x3D3,
                "vector_length": 0x3D4,
                "nr_iters": 0x3D5,
                "mode": 0x3D6,
            },
            0x3C0,
            0x3C3,
        )

    @staticmethod
    def lower_acc_barrier(acc_op: acc.AcceleratorOp) -> Sequence[Operation]:
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
            # Three nops for random but important reasons
            llvm.InlineAsmOp("nop", "", [], [], has_side_effects=True),
            llvm.InlineAsmOp("nop", "", [], [], has_side_effects=True),
            llvm.InlineAsmOp("nop", "", [], [], has_side_effects=True),
        ]

    @staticmethod
    def lower_acc_launch(acc_op: acc.AcceleratorOp) -> Sequence[Operation]:
        return [
            addr_val := arith.Constant(acc_op.launch_addr),
            val := arith.Constant(builtin.IntegerAttr.from_int_and_width(0, 5)),
            llvm.InlineAsmOp(
                "csrw $0, $1",
                # I = any 12 bit immediate, K = any 5 bit immediate
                # The K allows LLVM to emit an `csrrwi` instruction,
                # which has room for one 5 bit immediate only.
                "I, K",
                [addr_val, val],
                has_side_effects=True,
            ),
        ]

    @staticmethod
    def lower_setup_op(
        setup_op: acc.SetupOp, acc_op: acc.AcceleratorOp
    ) -> Sequence[Operation]:
        field_to_csr = dict(acc_op.field_items())
        ops: Sequence[Operation] = []
        for field, val in setup_op.iter_params():
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
