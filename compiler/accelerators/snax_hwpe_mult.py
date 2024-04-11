from collections.abc import Sequence

from xdsl.dialects import arith, builtin, linalg, memref
from xdsl.ir import Operation, SSAValue

from compiler.accelerators.snax import SNAXAcceleratorInterface
from compiler.dialects import acc


class SNAXHWPEMultAcceleratorInterface(SNAXAcceleratorInterface):
    """
    Accelerator Interface class for SNAX HWPE multiplier accelerator
    CSR lowerings are inherited from SNAXAcceleratorInterface.
    """

    name = "snax_hwpe_mult"
    fields = ("A", "B", "O", "vector_length", "nr_iters", "mode")

    def generate_setup_vals(
        self, op: linalg.Generic
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
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
