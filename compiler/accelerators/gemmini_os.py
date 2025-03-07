from typing import Sequence
from compiler.accelerators.gemmini import GemminiAccelerator
from compiler.dialects import accfg
from xdsl.dialects import linalg, arith, scf, builtin, memref
from xdsl.ir import Operation, SSAValue, Attribute, Block
from abc import ABC, abstractmethod
from compiler.accelerators.rocc import create_pairs, combine_pairs_to_ops


class GemminiConstants:
    """
    constants from include/gemmini_params.h
    """

    DIM = 16
    ADDR_LEN = 32
    BANK_NUM = 4
    BANK_ROWS = 4096
    ACC_ROWS = 1024
    MAX_BYTES = 64
    MAX_BLOCK_LEN = MAX_BYTES // (DIM * 1)
    MAX_BLOCK_LEN_ACC = MAX_BYTES // (DIM * 4)
    MVIN_SCALE_IDENTITY = 1.0
    ACC_SCALE_IDENTITY = 1.0
    ACC_SCALE_EXP_BITS = 8
    ACC_SCALE_SIG_BITS = 24


class GemminiOsAcceleratorBase(GemminiAccelerator, ABC):
    """
    Abstract base class for Gemmini accelerator instances in
    output stationary mode
    """

    def get_setup_op(self, vals: Sequence[Operation | SSAValue]) -> accfg.SetupOp:
        return accfg.SetupOp(vals, self.fields, self.name)

    def get_launch_await_seq(
        self, launch_vals: Sequence[Operation | SSAValue], state: accfg.SetupOp
    ) -> tuple[accfg.LaunchOp, accfg.AwaitOp]:
        ops = (
            token := accfg.LaunchOp(launch_vals, self.launch_fields, state),
            accfg.AwaitOp(token),
        )
        return ops


class GemminiMvinAccelerator(GemminiOsAcceleratorBase):
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


class GemminiMvoutAccelerator(GemminiOsAcceleratorBase):
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


class GemminiExAccelerator(GemminiOsAcceleratorBase):
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
        "k_COMPUTE.rs1": 4,  # and 5, both COMPUTE_PRELOADED and COMPUTE_ACCUMULATE
        "k_COMPUTE.rs2": 4,  # and 5, both COMPUTE_PRELOADED and COMPUTE_ACCUMULATE
        "is_accum": -1,
    }

    def get_conditional_launch_seq(
        self,
        launch_vals: Sequence[Operation | SSAValue],
        input_state: accfg.SetupOp,
        condition: SSAValue | Operation,
    ):
        """
        This launch sequence is special, because it will conditionally
        use a different opcode.

        (╯°□°)╯︵ ┻━┻

        if k == 0
          use op-code defined by k_COMPUTE_PRELOADED (4)
        else
          use op-code defined by k_COMPUTE_ACCUMULATE (5)

        """
        launch = accfg.LaunchOp(
            [*launch_vals, condition], self.launch_fields, input_state
        )
        return launch, accfg.AwaitOp(launch)

    @staticmethod
    def lower_acc_launch(
        launch_op: accfg.LaunchOp, acc_op: accfg.AcceleratorOp
    ) -> Sequence[Operation]:
        """
        Patched lower_acc_launch method to fix override in opcode emission

        (╥‸╥)

        LaunchOps that have override_gemmini_opcode attribute have to
        ignore acceleratorop for code emission
        """
        xcustom_acc = 3  # hardcoded to 3 for now

        vals = create_pairs(launch_op)

        mapping = acc_op.launch_field_items()

        if launch_op.attributes["override_gemmini_opcode"] is not None:
            repacked_mapping = dict(mapping)

            # override "k_COMPUTE"
            assert "k_COMPUTE.rs2" in repacked_mapping
            repacked_mapping["k_COMPUTE.rs2"] = launch_op.attributes[
                "override_gemmini_opcode"
            ]
            assert "k_COMPUTE.rs1" in repacked_mapping
            repacked_mapping["k_COMPUTE.rs1"] = launch_op.attributes[
                "override_gemmini_opcode"
            ]

            mapping = repacked_mapping.items()

        # Create the sequence of all operations that need to be emitted
        return combine_pairs_to_ops(mapping, vals, xcustom_acc)


def convert_to_accfg_sequence(op: linalg.Generic) -> Sequence[Operation]:
    """
    Convert a linalg generic to a sequence of different accelerator calls for
    gemmini output stationary mode
    """
    mvin = GemminiMvinAccelerator()
    ex = GemminiExAccelerator()
    mvout = GemminiMvoutAccelerator()
    ops_to_insert: Sequence[SSAValue | Operation] = []
    int_t = builtin.i64

    # TODO:
    #  - find A, B, C, A_row_stride, B_row_stride, C_row_stride
    #         I, J, K, pad_I, pad_J, pad_K,
    #  - assume: scale_factor = 1.0
    #  - define A_blocks, B_blocks
    #  - define {A,B,C}_sp_addr_start

    """
    linalg.generic {indexing_maps = [#map2, #map3, #map4, #map4, #map5], library_call="gemmini_os", iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview, %subview_0, %c0_i32, %c0_i32 : memref<?x128xi8, strided<[128, 1], offset: ?>>, memref<128x?xi8, strided<[128, 1], offset: ?>>, i32, i32) outs(%subview_1 : memref<?x?xi32, strided<[128, 1], offset: ?>>) {
    ^bb0(%in: i8, %in_2: i8, %in_3: i32, %in_4: i32, %out: i32):
        %2 = arith.extsi %in : i8 to i32
        %3 = arith.subi %2, %in_3 : i32
        %4 = arith.extsi %in_2 : i8 to i32
        %5 = arith.subi %4, %in_4 : i32
        %6 = arith.muli %3, %5 : i32
        %7 = arith.addi %out, %6 : i32
        linalg.yield %7 : i32
    }
    """
    # get A, B and C memrefs
    A, B, _, _, C = op.operands  # Don't use zero point adjustments
    ops_to_insert = []
    # pointers to data of A, B, C
    pointer_values = []
    # values of A_row_stride, B_row_stride, C_row_stride
    stride_values = []
    for operand in A, B, C:
        ops_to_insert.extend(
            [
                metadata := memref.ExtractStridedMetaDataOp(operand),
                pointer := memref.ExtractAlignedPointerAsIndexOp.get(operand),
                offset_ptr := arith.Addi(pointer, metadata.offset),
                offset_ptr_i64 := arith.IndexCastOp(
                    offset_ptr, int_t
                ),  # TODO: shouldn't this be i32?
                # Only add stride at index 0 for our experiments
                stride_i64 := arith.IndexCastOp(metadata.strides[0], int_t),
            ]
        )
        pointer_values.append(offset_ptr_i64.result)
        stride_values.append(stride_i64.result)

    # values of I, J, K
    size_values = []
    ops_to_insert.append(
        cst_16 := arith.Constant.from_int_and_width(16, builtin.IndexType()),
    )
    for operand, i in zip([A, B, A], [0, 1, 1]):
        ops_to_insert.extend(
            [
                metadata := memref.ExtractStridedMetaDataOp(operand),
                divided_size := arith.DivUI(metadata.sizes[i], cst_16),
                size_i64 := arith.IndexCastOp(divided_size, int_t),
            ]
        )
        size_values.append(size_i64.result)

    # We can assume padding to be 0 as we always have multiples of 16 for sizes
    ops_to_insert.append(cst_0 := arith.Constant.from_int_and_width(0, int_t))
    padding_values = [cst_0.result, cst_0.result, cst_0.result]

    ## now we have:
    # A, B, C               as memrefs in vars A, B, C
    # A, B, C               as pointers in pointer_values
    # {A,B,C}_row_stride    in stride_values
    # I, J, K               in size_values
    # {I,J,K}_pad           in padding_values
    I, J, K = size_values
    I_pad, J_pad, K_pad = padding_values

    ## Now we calculate {A,B}_blocks:
    """
    const int A_blocks = K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN;
    const int B_blocks = J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN;
    """
    ops_to_insert.extend(
        [
            max_block_len := arith.Constant.from_int_and_width(
                GemminiConstants.MAX_BLOCK_LEN, int_t
            ),
            A_blocks := arith.MinSI(max_block_len, K),
            B_blocks := arith.MinSI(max_block_len, J),
        ]
    )

    blocks_values = [A_blocks, B_blocks]
    ## now we calc {A,B,C}_sp_addr_start
    A_sp_addr_start = cst_0.result
    # const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
    ops_to_insert.extend(
        (
            dim := arith.Constant.from_int_and_width(GemminiConstants.DIM, int_t),
            num_t_rows := arith.Constant.from_int_and_width(
                GemminiConstants.BANK_NUM * GemminiConstants.BANK_ROWS, int_t
            ),
            K_t_J := arith.Muli(K, J),
            t_dim := arith.Muli(dim, K_t_J),
            B_sp_addr_start := arith.Subi(num_t_rows, t_dim),
        )
    )
    # const uint32_t C_sp_addr_start = (3 << (ADDR_LEN-2)) | (full_C << (ADDR_LEN-3));
    # We know that full_C = false
    ops_to_insert.append(
        C_sp_addr_start := arith.Constant.from_int_and_width(
            (3 << (GemminiConstants.ADDR_LEN - 2)), int_t
        )
    )
    # now we invoke mvinb
    B_row_stride = stride_values[1]
    B_ptr = pointer_values[1]
    ops_to_insert.extend(
        gen_move_in_b(
            B_ptr,
            B_row_stride,
            B_blocks,
            B_sp_addr_start,
            J,
            K,
            J_pad,
            K_pad,
            int_t=int_t,
        )
    )

    # now we invoke mvinA repeatedly
    """
    // Move-in A
    gemmini_extended_config_ld(A_row_stride * sizeof(elem_t), A_scale_factor);
    for (size_t i = 0; i < I; i++) {
        for (size_t k = 0; k < K; k += A_blocks) {
        const elem_t * const A_dram_addr = A + (i*A_row_stride + k)*DIM;
        const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
        const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
        const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
        const size_t rows = DIM - (i == I-1 ? pad_I : 0);
        gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
        }
    }
    """

    # And now the main boo-hoo:
    """
    for (size_t i = 0; i < I; i++) {
        for (size_t j = 0; j < J; j++) {
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        for (size_t k = 0; k < K; k++) {

            const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
            const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;

            uint32_t out_sp_addr = k == K-1 ? C_sp_addr : GARBAGE_ADDR;

            // If we're not using a bias, then we want to overwrite what's in the
            // accumulator, rather than writing over it
            int no_bias_new_matrix = no_bias && D != NULL && k == K-1;
            if (no_bias_new_matrix) {
            out_sp_addr &= ~(1 << (ADDR_LEN-2));
            }

            const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
            const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
            const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
            const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
            const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
            const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

            gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);

            if (k == 0) { // First iteration
            gemmini_extended_compute_preloaded(A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            } else { // All other iterations
            gemmini_extended_compute_accumulated(A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            }
        }
        }
    }
    """

    # finally, move D out
    """
    // Move-out C
    const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

    for (size_t i = 0; i < I; i++) {
        for (size_t j = 0; j < J; j++) {
            void * const C_dram_addr = (int8_t*)C + (i*C_row_stride + j)*DIM*sizeof_C;
            const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

            const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
            const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

            gemmini_extended_mvout(C_dram_addr, C_sp_addr, C_cols, C_rows);
        }
    }
    """

    ops_to_insert.extend(
        [
            zero := arith.Constant.from_int_and_width(0, 64),
            zero_bit := arith.Constant.from_int_and_width(0, 1),
            setup := mvin.get_setup_op([zero, zero]),
            *mvin.get_launch_await_seq([zero, zero], setup),
            setup := ex.get_setup_op([zero, zero]),
            *ex.get_conditional_launch_seq([zero, zero, zero, zero], setup, zero_bit),
            setup := mvout.get_setup_op([zero, zero]),
            *mvout.get_launch_await_seq([zero, zero], setup),
        ]
    )
    return ops_to_insert


def add(x: SSAValue | Operation, y: SSAValue | Operation) -> Operation:
    return arith.Addi(x, y)


def mul(x: SSAValue | Operation, y: SSAValue | Operation) -> Operation:
    return arith.Muli(x, y)


def leq(x: SSAValue | Operation, y: SSAValue | Operation) -> Operation:
    """compare x <= y"""
    return arith.Cmpi(x, y, "sle")


def eq(x: SSAValue | Operation, y: SSAValue | Operation) -> Operation:
    """compare x <= y"""
    return arith.Cmpi(x, y, "eq")


def select(
    cond: SSAValue | Operation, x: SSAValue | Operation, y: SSAValue | Operation
) -> Operation:
    """cond ? x : y"""
    return arith.Select(cond, x, y)


def sub(x, y) -> Operation:
    return arith.Subi(x, y)


def gen_move_in_b(
    B, B_row_stride, B_blocks, B_sp_addr_start, J, K, J_pad, K_pad, *, int_t: Attribute
) -> Sequence[Operation]:
    """
    Generate the following C equivalent:

    // Move-in B
    gemmini_extended_config_ld(B_row_stride * sizeof(elem_t), B_scale_factor);
    for (size_t j = 0; j < J; j += B_blocks) {
        for (size_t k = 0; k < K; k++) {
        const elem_t * const B_dram_addr = B + (k*B_row_stride + j)*DIM;
        const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;
        const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
        const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
        const size_t rows = DIM - (k == K-1 ? pad_K : 0);
        gemmini_extended_mvin(B_dram_addr, B_sp_addr, cols, rows);
        }
    }
    """
    from xdsl.builder import ImplicitBuilder

    ops = [
        cst_0 := arith.Constant.from_int_and_width(0, int_t),
        cst_1 := arith.Constant.from_int_and_width(1, int_t),
        DIM := arith.Constant.from_int_and_width(GemminiConstants.DIM, int_t),
        float_one_as_int := arith.Constant.from_int_and_width(0x3F800000, int_t),
    ]
    calc_ops, rs1 = build_mv_setup_rs1(float_one_as_int)
    ops.extend(calc_ops)
    ops.append(
        cfg_op := GemminiMvinAccelerator().get_setup_op(
            (
                rs1,
                B_row_stride,
            )
        )
    )

    outer = Block(arg_types=[int_t])
    with ImplicitBuilder(outer) as (j,):
        inner = Block(arg_types=[int_t])
        with ImplicitBuilder(inner) as (k,):
            # B_dram_addr = B + (k*B_row_stride + j)*DIM;
            B_dram_addr = add(B, mul(add(mul(k, B_row_stride), j), DIM))
            # B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;
            B_sp_addr = add(B_sp_addr_start, mul(add(mul(k, j), j), DIM))
            # blocks = j + B_blocks <= J ? B_blocks : J-j;
            blocks = select(
                leq(add(j, B_blocks), J),  # j+B_blocks <= J
                B_blocks,
                sub(J, j),
            )
            # cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
            # _intermediate = select(leq(add(j, blocks), J), J_pad, cst_0)
            _intermediate = cst_0  # padding is always 0
            cols = sub(mul(blocks, DIM), _intermediate)
            # rows = DIM - (k == K-1 ? pad_K : 0);
            # _intermediate2 = select(eq(k, sub(K, cst_1)), K_pad, cst_0)
            _intermediate2 = cst_0  # padding is always 0
            rows = sub(DIM, _intermediate2)
            # gemmini_extended_mvin(B_dram_addr, B_sp_addr, cols, rows);
            GemminiMvinAccelerator().get_launch_await_seq(
                (B_dram_addr, build_mv_launch_rs2(B_sp_addr, cols, rows)),
                cfg_op.out_state,
            )
            scf.Yield()
        scf.For(cst_0, K, cst_1, [], body=inner)
        scf.Yield()

    return (*ops, scf.For(cst_0, J, B_blocks, [], body=outer))


def build_mv_launch_rs2(
    sp_addr: SSAValue, cols: SSAValue, rows: SSAValue
) -> tuple[list[Operation], SSAValue, SSAValue]:
    """
    Do the value packing for the mvin/mvout instructions:

    rs2 = ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr)
    """
    ADDR_LEN = arith.Constant.from_int_and_width(GemminiConstants.ADDR_LEN, builtin.i64)
    ADDR_LEN_P_16 = arith.Constant.from_int_and_width(
        GemminiConstants.ADDR_LEN + 16, builtin.i64
    )
    vint = arith.OrI(arith.ShLI(rows, ADDR_LEN_P_16), sp_addr)
    rval = arith.OrI(vint, arith.ShLI(cols, ADDR_LEN))
    return rval.result


def build_mv_setup_rs1(scale_factor) -> tuple[Sequence[Operation], SSAValue]:
    """
    build this giant mistake:

    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
        ((uint64_t)(scale_t_to_scale_t_bits(scale)) << 32)
        | ((uint64_t)(block_mvin_stride) << 16)
        | ((uint64_t)(pixel_repeats) << 8)
        | ((id) << 3)
        | ((shrunk) << 2)
        | CONFIG_LD,
        stride,
        k_CONFIG
    )

    default values:
        scale = 1.0
        shrunk = false (0)
        block_mvin_srtride = DIM
        pixel_repeats = 1
        id = 0

    and CONFIG_LD = 1
    """
    return (
        cst_32 := arith.Constant.from_int_and_width(32, builtin.i64),
        v0 := arith.ShLI(scale_factor, cst_32),
        other_consts := arith.Constant.from_int_and_width(
            (GemminiConstants.DIM << 16)  # block_mvin_srtride
            | (1 << 8)  # pixel_repeats
            | (2),  # CONFIG_LD
            builtin.i64,
        ),
        res := arith.OrI(v0, other_consts),
    ), res.result
