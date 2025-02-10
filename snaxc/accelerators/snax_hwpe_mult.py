from collections.abc import Sequence

from xdsl.dialects import arith, builtin, linalg, memref
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.utils.hints import isa

from snaxc.accelerators.snax import SNAXAccelerator, SNAXPollingBarrier
from snaxc.dialects import accfg


class SNAXHWPEMultAccelerator(SNAXAccelerator, SNAXPollingBarrier):
    """
    Accelerator Interface class for SNAX HWPE multiplier accelerator
    CSR lowerings are inherited from SNAXAcceleratorInterface.
    """

    name = "snax_hwpe_mult"
    fields = ("A", "B", "O", "vector_length", "nr_iters", "mode")
    launch_fields = ("launch",)

    def convert_to_acc_ops(self, op: linalg.GenericOp) -> Sequence[Operation]:
        """
        Lowers the operation op to a sequence of acc_ops.
        acc_ops are:
            - *.op that generates SSAValues consumed by accfg.setup
            - accfg.setup
            - accfg.launch
            - accfg.await
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
            launch_val := arith.ConstantOp(
                builtin.IntegerAttr.from_int_and_width(0, 5)
            ),
            token := accfg.LaunchOp([launch_val], self.launch_fields, setup),
            accfg.AwaitOp(token),
        ]

    def _generate_setup_vals(
        self, op: linalg.GenericOp
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        """
        Produce a `Sequence[Operation], SSAValue` tuple
        for each field that contains:

        - a list of operations that calculate the field value
        - a reference to the SSAValue containing the calculated field value
        """
        a, b, c = op.operands
        for operand in op.operands:
            assert isa(operand.type, builtin.MemRefType[Attribute])

        zero = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
        iters_one = arith.ConstantOp.from_int_and_width(1, 32)
        mode_one = arith.ConstantOp.from_int_and_width(1, 32)
        dim = memref.DimOp.from_source_and_index(a, zero)
        dim_i32 = arith.IndexCastOp(dim, builtin.i32)
        vector_length = [zero, dim, dim_i32], dim_i32.result

        nr_iters = [iters_one], iters_one.result
        mode = [mode_one], mode_one.result

        ptrs = [
            (
                [
                    ptr := memref.ExtractAlignedPointerAsIndexOp.get(ref),
                    metadata := memref.ExtractStridedMetaDataOp(ref),
                    el_bytes := arith.ConstantOp.from_int_and_width(
                        ref.type.element_type.size, builtin.IndexType()
                    ),
                    byte_offset := arith.MuliOp(metadata.offset, el_bytes),
                    ptr_plus_byte_offset := arith.AddiOp(
                        ptr, byte_offset, builtin.IndexType()
                    ),
                    ptr_i32 := arith.IndexCastOp(ptr_plus_byte_offset, builtin.i32),
                ],
                ptr_i32.result,
            )
            for ref in (a, b, c)
        ]

        return ptrs + [nr_iters] + [vector_length] + [mode]

    def generate_acc_op(self) -> accfg.AcceleratorOp:
        """
        Return this accelerator op:

        "accfg.accelerator"() <{
            name            = @snax_hwpe_mult,
            fields          = {A=0x3d0, B=0x3d1, O=0x3d3, n_iters=0x3d4,
                               vector_length=0x3d5, mode=0x3d6},
            launch_fields   = {"launch"=0x3c0},
            barrier = 0x3c3,
        }> : () -> ()
        """
        return accfg.AcceleratorOp(
            self.name,
            {
                "A": 0x3D0,
                "B": 0x3D1,
                "O": 0x3D3,
                "vector_length": 0x3D4,
                "nr_iters": 0x3D5,
                "mode": 0x3D6,
            },
            {"launch": 0x3C0},
            0x3C3,
        )
