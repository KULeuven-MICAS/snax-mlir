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
            0x0BAD,  # Gemmini has no separate launch instruction, but appears to launch on k_LOOP_WS configuration
            0x0BAD,  # Gemmini works appears to work synchronously, and does not have a separate await instruction
        )

    # static void sp_tiled_matmul_ws(const elem_t * A, const elem_t * B,
    # const void * D, void * C,
    # scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
    # size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
    # size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
    # bool a_transpose, bool b_transpose,
    # bool full_C, bool low_D,
    # bool no_bias, bool repeating_bias,
    # int act) {


# // weight-stationary matmul loop
##define gemmini_loop_ws(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_stride, B_stride, D_stride, C_stride, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act) \
#  { \
#    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
#    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
#    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
#    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A_stride, B_stride, k_LOOP_WS_CONFIG_STRIDES_AB) \
#    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D_stride, C_stride, k_LOOP_WS_CONFIG_STRIDES_DC) \
#    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
#  }
