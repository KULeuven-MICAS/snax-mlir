from collections.abc import Sequence

from xdsl.dialects import arith, builtin, linalg, memref
from xdsl.ir import Operation, SSAValue

from compiler.accelerators.snax import SNAXAccelerator, SNAXPollingBarrier2
from compiler.dialects import accfg


class SNAXGEMMAccelerator(SNAXAccelerator, SNAXPollingBarrier2):
    """
    Accelerator Interface class for SNAX GEMM accelerator
    CSR lowerings are inherited from SNAXAcceleratorInterface.

    Based on:
    snax_cluster/target/snitch_cluster/sw/snax/gemm/src/snax-gemm-lib.c
    """

    name = "snax_gemm"
    fields = (
        "size_setting",
        "local_a",
        "local_b",
        "local_c",
        "strideInnermostA",
        "strideInnermostB",
        "strideInnermostC",
        "ldA",
        "ldB",
        "ldC",
        "strideA",
        "strideB",
        "strideC",
        "subtractions",
    )
    launch_fields = ("launch",)

    def generate_acc_op(self) -> accfg.AcceleratorOp:
        """
        Return a SNAX GEMM accelerator op with some default field adresses
        """
        return accfg.AcceleratorOp(
            self.name,
            {
                "size_setting": 0x3C0,
                "local_a": 0x3C1,
                "local_b": 0x3C2,
                "local_c": 0x3C3,
                "strideInnermostA": 0x3C4,
                "strideInnermostB": 0x3C5,
                "strideInnermostC": 0x3C6,
                "ldA": 0x3C7,
                "ldB": 0x3C8,
                "ldC": 0x3C9,
                "strideA": 0x3CA,
                "strideB": 0x3CB,
                "strideC": 0x3CC,
                "subtractions": 0x3CE,
            },
            {"launch": 0x3CF},
            0x3CF,
        )

    def convert_to_acc_ops(self, op: linalg.Generic) -> Sequence[Operation]:
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
            token := accfg.LaunchOp([launch_val], self.launch_fields, setup),
            accfg.AwaitOp(token),
        ]

    def _generate_setup_vals(
        self, op: linalg.Generic
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        """
        Produce a `Sequence[Operation], SSAValue` tuple
        for each field that contains:

        - a list of operations that calculate the field value
        - a reference to the SSAValue containing the calculated field value
        """

        def _generate_subtract_config(subtraction_a: SSAValue, subtraction_b: SSAValue):
            """
            Helper function to calculate a subtraction configuration for SNAX GEMM

            Should be equivalent to C code:

            int32_t gen_subtraction_config(int8_t subtraction_a, int8_t subtraction_b) {
                return ((uint8_t)subtraction_b << 8) | (uint8_t)subtraction_a;
            }

            Instead of casting to 8-bit unsinged integer, we use AND with 0xFF.
            This prevents a possible overflow on shifting the values.
            """
            return [
                c_8 := arith.Constant.from_int_and_width(8, 32),
                c_0xFF := arith.Constant.from_int_and_width(0xFF, 32),
                subtraction_a_no_overflow := arith.AndI(subtraction_a, c_0xFF),
                subtraction_b_no_overflow := arith.AndI(subtraction_b, c_0xFF),
                shifted_subtraction_b := arith.ShLI(subtraction_b_no_overflow, c_8),
                or_op := arith.OrI(subtraction_a_no_overflow, shifted_subtraction_b),
            ], or_op.result

        def _generate_size_config(batch: int, m: int, k: int, n: int):
            """
            Helper function to calculate the size config for SNAX GEMM
            Should be equivalent to C code:

            int32_t gen_size_config(uint8_t Batch, uint8_t M, uint8_t K, uint8_t N) {
                return ( (int32_t)Batch << 24) | ( (int32_t)M << 16) | ( (int32_t)K << 8) | (int32_t)N;
            }
            """
            return [
                batch_i8 := arith.Constant.from_int_and_width(batch, 8),
                m_i8 := arith.Constant.from_int_and_width(m, 8),
                k_i8 := arith.Constant.from_int_and_width(k, 8),
                n_i8 := arith.Constant.from_int_and_width(n, 8),
                c_8 := arith.Constant.from_int_and_width(8, 32),
                c_16 := arith.Constant.from_int_and_width(16, 32),
                c_24 := arith.Constant.from_int_and_width(24, 32),
                # Perform zero extension, as these ORed together later
                batch_i32 := arith.ExtUIOp(batch_i8, builtin.i32),
                m_i32 := arith.ExtUIOp(m_i8, builtin.i32),
                k_i32 := arith.ExtUIOp(k_i8, builtin.i32),
                n_i32 := arith.ExtUIOp(n_i8, builtin.i32),
                k_shift := arith.ShLI(k_i32, c_8),
                m_shift := arith.ShLI(m_i32, c_16),
                batch_shift := arith.ShLI(batch_i32, c_24),
                or_n_k_op := arith.OrI(k_shift, n_i32),
                or_m_or_x_op := arith.OrI(m_shift, or_n_k_op),
                or_batch_or_x_op := arith.OrI(batch_shift, or_m_or_x_op),
            ], or_batch_or_x_op.result

        def _get_constants(values, width):
            return [
                (
                    [const := arith.Constant.from_int_and_width(value, width)],
                    const.result,
                )
                for value in values
            ]

        a, b, zpa, zpb, c = op.operands

        constants = [256, 256, 256, 512, 512, 512, 0, 0, 0]

        size_config = _generate_size_config(1, 2, 2, 2)

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

        return [
            size_config,
            *ptrs,
            *_get_constants(constants, 32),
            _generate_subtract_config(zpa, zpb)
        ]
