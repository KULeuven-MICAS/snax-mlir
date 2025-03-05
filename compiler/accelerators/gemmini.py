from collections.abc import Sequence

from xdsl.dialects import arith, linalg, memref
from xdsl.dialects.builtin import IndexType, i64
from xdsl.ir import Operation

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

    def _gemmini_loop_ws(
        self,
        pad_K,
        pad_J,
        pad_I,
        K,
        J,
        I,
        A,
        B,
        D,
        C,
        A_stride,
        B_stride,
        D_stride,
        C_stride,
    ):
        # Make lists for constructors of accfg ops
        ops_to_insert = []
        ops_to_configure = []
        ops_to_launch = []
        # Insert hardcoded constants for now
        constants = [
            [pad_K, pad_J, pad_I],  # pad_K, pad_J, pad_I
            [K, J, I],  # K,J,I
            [A],  # A,B
            [B],
            [D],  # D,C
            [C],
            [A_stride],  # A_stride, B_stride
            [B_stride],
            [D_stride],  # D_stride, C_stride
            [C_stride],
        ]
        shifts = [
            [32, 16, 0],  # pad_K, pad_J, pad_I
            [32, 16, 0],  # K,J,I
            [0],  # A,B
            [0],
            [0],  # D,C
            [0],
            [0],  # A_stride, B_stride
            [0],
            [0],  # D_stride, C_stride
            [0],
        ]
        # Pack bits together if necessary for configuration ops
        for shift_vals, constant_vals in zip(shifts, constants):
            ops = list(pack_bitlist(values=constant_vals, offsets=shift_vals, dtype=64))
            ops_to_insert.extend(ops)
            ops_to_configure.extend(ops[-1].results)

        # Insert hardcoded constants for now
        launch_constants = [
            [0, 0, 0],  # act, low_D, full_C
            [0, 0],  # A_transpose, B_transpose
        ]
        launch_shifts = [
            [2, 1, 0],
            [1, 0],
        ]

        # Pack bits together for launch ops
        for shift_vals, constant_vals in zip(launch_shifts, launch_constants):
            ops = list(pack_bitlist(values=constant_vals, offsets=shift_vals, dtype=64))
            ops_to_insert.extend(ops)
            ops_to_launch.extend(ops[-1].results)

        return [
            *ops_to_insert,
            setup := accfg.SetupOp(ops_to_configure, self.fields, self.name),
            token := accfg.LaunchOp(ops_to_launch, self.launch_fields, setup),
            accfg.AwaitOp(token),
        ]

    def convert_to_acc_ops(self, op: linalg.Generic) -> Sequence[Operation]:
        a, b, _, _, c = op.operands  # Don't use zero point adjustments
        ops_to_insert = []
        pointer_values = []
        stride_values = []
        for operand in a, b, c:
            ops_to_insert.extend(
                [
                    metadata := memref.ExtractStridedMetaDataOp(operand),
                    pointer := memref.ExtractAlignedPointerAsIndexOp.get(operand),
                    offset_ptr := arith.Addi(pointer, metadata.offset),
                    offset_ptr_i64 := arith.IndexCastOp(offset_ptr, i64),
                    # Only add stride at index 0 for our experiments
                    stride_i64 := arith.IndexCastOp(metadata.strides[0], i64),
                ]
            )
            pointer_values.append(offset_ptr_i64.result)
            stride_values.append(stride_i64.result)

        size_values = []
        for operand, i in zip([a, b, a], [0, 1, 1]):
            ops_to_insert.extend(
                [
                    cst_16 := arith.Constant.from_int_and_width(16, IndexType()),
                    metadata := memref.ExtractStridedMetaDataOp(operand),
                    divided_size := arith.DivUI(metadata.sizes[i], cst_16),
                    size_i64 := arith.IndexCastOp(divided_size, i64),
                ]
            )
            size_values.append(size_i64.result)

        ops_to_insert.extend(
            [
                cst_0 := arith.Constant.from_int_and_width(0, 64),
                *self._gemmini_loop_ws(
                    cst_0,
                    cst_0,
                    cst_0,
                    size_values[2],  # K
                    size_values[1],  # J
                    size_values[0],  # I
                    pointer_values[0],  # a
                    pointer_values[1],  # b
                    cst_0,  # d
                    pointer_values[2],  # c
                    stride_values[0],  # a
                    stride_values[1],  # b
                    stride_values[2],  # d --> use same as C
                    stride_values[2],  # c
                ),
            ]
        )

        return ops_to_insert


class GemminiMvinAccelerator(GemminiAccelerator):
    name = "gemmini_mvin"
    """
    For some weird reason, all of these config instructions use the same 
    k_CONFIG opcode, but they use one of rs's contents to switch register sets.
    Also, there are actually 3 data movers, the docs say this is to use them
    in parallel, but they are not programmed in parallel in the OS kernel?
    Hence the name k_CONFIG (opcode) CONFIG_LD (modifier for loading/mvin)
    and id0 (modifier for which of 3 movers is used).

    Note: You need to manually make sure: 
    * that id is programmed to cst=0!
    * that CONFIG_LD modifier is set for k_CONFIG opcode
    """
    fields = {
        "k_CONFIG_k_CONFIG_LD_id0.rs1": 0,
        "k_CONFIG_k_CONFIG_LD_id0.rs2": 0,
    }
    launch_fields = {
        "k_MVIN.rs1": 2,
        "k_MVIN.rs2": 2,
    }


class GemminiMvoutAccelerator(GemminiAccelerator):
    name = "gemmini_mvout"
    """
    For some weird reason, all of these config instructions use the same 
    k_CONFIG opcode, but they use one of rs's contents to switch registers sets.
    Hence the name k_CONFIG (opcode) CONFIG_ST (modifier for storing/mvout)

    Note: You need to manually make sure: 
    * that CONFIG_ST modifier is set for k_CONFIG opcode
    """
    fields = {
        "k_CONFIG_k_CONFIG_ST.rs1": 0,
        "k_CONFIG_k_CONFIG_ST.rs2": 0,
    }
    launch_fields = {
        "k_MVOUT.rs1": 3,
        "k_MVOUT.rs2": 3,
    }

class GemminiExAccelerator(GemminiAccelerator):
    name = "gemmini_ex"
    """
    For some weird reason, all of these config instructions use the same 
    k_CONFIG opcode, but they use one of rs's contents to switch registers sets.
    Hence the name k_CONFIG (opcode) CONFIG_EX (modifier for execution)

    Note: You need to manually make sure: 
    * that CONFIG_EX modifier is set for k_CONFIG opcode
    * the launch fields of the final instruction in this sequence are the same,
      but the compiler needs to emit both accumulating and non-accumulating
      instructions, because this is based on the op-code of the operation.
    """
    fields = {
        "k_CONFIG_k_CONFIG_EX.rs1": 0,
        "k_CONFIG_k_CONFIG_EX.rs2": 0,
    }
    launch_fields = {
        "k_PRELOAD.rs1": 2,
        "k_PRELOAD.rs2": 2,
    }
