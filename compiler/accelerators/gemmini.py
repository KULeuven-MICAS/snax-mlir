from compiler.accelerators.rocc import RoCCAccelerator
from compiler.dialects import accfg


class GemminiAccelerator(RoCCAccelerator):
    name = "gemmini"

    fields = {
        "k_LOOP_WS_CONFIG_BOUNDS.rs1": 9,
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs1": 10,
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs1": 11,
        "k_LOOP_WS_CONFIG_STRIDES_AB.rs1": 12,
        "k_LOOP_WS_CONFIG_STRIDES_DC.rs1": 13,
        "k_LOOP_WS_CONFIG_BOUNDS.rs2": 9,
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs2": 10,
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs2": 11,
        "k_LOOP_WS_CONFIG_STRIDES_AB.rs2": 12,
        "k_LOOP_WS_CONFIG_STRIDES_DC.rs2": 13,
    }

    launch_fields = {
        "k_LOOP_WS.rs1": 8,
        "k_LOOP_WS.rs2": 8,
    }

    def generate_acc_op(self) -> accfg.AcceleratorOp:
        """
        Return this accelerator op:

        "acc2.accelerator"() <{
            name            = @gemmini,
            fields          = {...},
            launch_fields   = {...},
            barrier         = {},
        }> : () -> ()
        """
        return accfg.AcceleratorOp(
            self.name,
            self.fields,
            self.launch_fields,  # Gemmini has no separate launch instruction,
            # but appears to launch on k_LOOP_WS configuration
            0x0BAD,  # Gemmini appears to work synchronously,
            # and does not have a separate await instruction
        )
