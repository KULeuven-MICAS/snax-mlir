from compiler.accelerators.rocc import RoCCAccelerator
from compiler.dialects import accfg
from compiler.util.pack_bitlist import pack_bitlist


class GemminiAccelerator(RoCCAccelerator):
    name = "gemmini"

    fields = {
        "k_LOOP_WS_CONFIG_BOUNDS.rs1": 9,
        "k_LOOP_WS_CONFIG_BOUNDS.rs2": 9,
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs1": 10,
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs2": 10,
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs1": 11,
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs2": 11,
        "k_LOOP_WS_CONFIG_STRIDES_AB.rs1": 12,
        "k_LOOP_WS_CONFIG_STRIDES_AB.rs2": 12,
        "k_LOOP_WS_CONFIG_STRIDES_DC.rs1": 13,
        "k_LOOP_WS_CONFIG_STRIDES_DC.rs2": 13,
    }

    launch_fields = {
        "k_LOOP_WS.rs1": 8,
        "k_LOOP_WS.rs2": 8,
    }

    def generate_acc_op(self) -> accfg.AcceleratorOp:
        """
        Return this accelerator op:

        "accfg.accelerator"() <{
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

    def _gemmini_loop_ws(self):
        # Make lists for constructors of accfg ops
        ops_to_insert = []
        ops_to_configure = []
        ops_to_launch = []
        # Insert hardcoded constants for now
        constants = [
            (2, 2, 2),  # pad_K, pad_J, pad_I
            (2, 2, 2),  # K,J,I
            (2),  # A,B
            (2),
            (2),  # D,C
            (2),
            (2),  # A_stride, B_stride
            (2),
            (2),  # D_stride, C_stride
            (2),
        ]
        shifts = [
            (32, 16, 0),  # pad_K, pad_J, pad_I
            (32, 16, 0),  # K,J,I
            (0),  # A,B
            (0),
            (0),  # D,C
            (0),
            (0),  # A_stride, B_stride
            (0),
            (0),  # D_stride, C_stride
            (0),
        ]
        # Pack bits together if necessary for configuration ops
        for shifts, constants in zip(shifts, constants):
            ops = list(pack_bitlist(values=constants, offsets=shifts, dtype=64))
            ops_to_insert.append(ops[:-1])
            ops_to_configure.append(ops[-1])

        # Insert hardcoded constants for now
        launch_constants = [
            (0, 0, 0),  # act, low_D, full_C
            (0, 0),  # A_transpose, B_transpose
        ]
        launch_shifts = [
            (2, 1, 0),
            (1, 0),
        ]

        # Pack bits together for launch ops
        for shifts, constants in zip(launch_shifts, launch_constants):
            ops = list(pack_bitlist(values=constants, offsets=shifts, dtype=64))
            ops_to_insert.append(ops[:-1])
            ops_to_launch.append(ops[-1])
        return [
            *ops_to_insert,
            setup := accfg.SetupOp(ops_to_configure, self.fields, self.name),
            token := accfg.LaunchOp(ops_to_launch, self.launch_fields, setup),
            accfg.AwaitOp(token),
        ]
