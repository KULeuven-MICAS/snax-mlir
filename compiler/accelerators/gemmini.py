from compiler.accelerators.rocc import RoCCAccelerator
from compiler.dialects import acc


class GemminiAccelerator(RoCCAccelerator):
    name = "gemmini"

    fields = {
        "k_LOOP_WS_CONFIG_BOUNDS.rs1": 9,
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs1": 10,
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs1": 11,
        "k_LOOP_WS_CONFIG_STRIDES_AB.rs1": 12,
        "k_LOOP_WS_CONFIG_STRIDES_DC.rs1": 13,
        "k_LOOP_WS.rs1": 8,
        "k_LOOP_WS_CONFIG_BOUNDS.rs2": 9,
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs2": 10,
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs2": 11,
        "k_LOOP_WS_CONFIG_STRIDES_AB.rs2": 12,
        "k_LOOP_WS_CONFIG_STRIDES_DC.rs2": 13,
        "k_LOOP_WS.rs2": 8,
    }

    def generate_acc_op(self) -> acc.AcceleratorOp:
        """
        Return this accelerator op:

        "acc2.accelerator"() <{
            name            = @gemmini,
            fields          = {A=0x3d0, B=0x3d1, O=0x3d3, n_iters=0x3d4,
                               vector_length=0x3d5, mode=0x3d6},
            launch_addr     = 0x3c0,
            barrier = 0x3c3,
        }> : () -> ()
        """
        return acc.AcceleratorOp(
            self.name,
            self.fields,
            0x0BAD,  # Gemmini has no separate launch instruction,
            # but appears to launch on k_LOOP_WS configuration
            0x0BAD,  # Gemmini works appears to work synchronously,
            # and does not have a separate await instruction
        )
