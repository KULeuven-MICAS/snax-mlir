builtin.module {
  "acc2.accelerator"() <{
      name            = @gemmini,
      fields = { k_LOOP_WS_CONFIG_BOUNDS.rs1=9, k_LOOP_WS_CONFIG_ADDRS_AB.rs1=10,
        k_LOOP_WS_CONFIG_ADDRS_DC.rs1=11,
        k_LOOP_WS_CONFIG_STRIDES_AB.rs1=12,
        k_LOOP_WS_CONFIG_STRIDES_DC.rs1=13,
        k_LOOP_WS_CONFIG_BOUNDS.rs2=9,
        k_LOOP_WS_CONFIG_ADDRS_AB.rs2=10,
        k_LOOP_WS_CONFIG_ADDRS_DC.rs2=11,
        k_LOOP_WS_CONFIG_STRIDES_AB.rs2=12,
        k_LOOP_WS_CONFIG_STRIDES_DC.rs2=13
        },
      launch_fields   = {
        k_LOOP_WS.rs1=8,
        k_LOOP_WS.rs2=8},
      barrier         = 0x0BAD
  }> : () -> ()
  func.func public @sp_tiled_matmul_ws(
    %A: !llvm.ptr, %B : !llvm.ptr, %D: !llvm.ptr, %C: !llvm.ptr, 
    %A_scale_factor: f64, %B_scale_factor: f64, %D_scale_factor: f64,
    %I: index , %J: index, %K: index,
    %pad_I: index, %pad_J: index, %pad_K: index,
    %A_row_stride: index, %B_row_stride: index, %D_row_stride: index, %C_row_stride: index,
    %a_transpose: i1, %b_transpose: i1, %full_C: i1, %low_D: i1,
    %no_bias: i1, %repeating_bias: i1, %act: i32,
    %a_spad_id: i32, %b_spad_id: i32
   ) {
    %t = arith.constant 32 : i64
    %c_32 = arith.constant 32 : index
    %c_18 = arith.constant 18 : index
    %c_16 = arith.constant 16 : index
    %c_8 = arith.constant 8 : index
    %c_2 = arith.constant 2 : index
    %c_1 = arith.constant 1 : index
    %false = arith.constant 0 : i1
    %true = arith.constant 1 : i1
    %NULL = arith.constant 0 : i64
    %NULLptr = "llvm.inttoptr"(%NULL) : (i64) -> !llvm.ptr
    

    // ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I)
    %pad_K_shift_32 = arith.shli %pad_K, %c_32 : index
    %pad_J_shift_16 = arith.shli %pad_J, %c_16 : index
    %pad_K_shift_or_pad_J_shift = arith.ori %pad_K_shift_32, %pad_J_shift_16 : index
    %pad_K_shift_or_pad_J_shift_or_pad_I_shift= arith.ori %pad_K_shift_or_pad_J_shift, %pad_I : index

    // ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I)
    %K_shift_32 = arith.shli %K, %c_32 : index
    %J_shift_16 = arith.shli %J, %c_16 : index
    %K_shift_or_J_shift = arith.ori %K_shift_32, %J_shift_16 : index
    %K_shift_or_J_shift_or_I_shift = arith.ori %pad_K_shift_or_pad_J_shift, %pad_J : index

    // no_bias ? NULL : D
    %D_selected = arith.select %no_bias, %NULLptr, %D : !llvm.ptr 

    // Convert to LLVM compatible types
    %pad_K_shift_or_pad_J_shift_or_pad_I_shift_i64 = "arith.index_cast"(%pad_K_shift_or_pad_J_shift_or_pad_I_shift) : (index) -> i64
    %K_shift_or_J_shift_or_I_shift_i64 = "arith.index_cast"(%K_shift_or_J_shift_or_I_shift) : (index) -> i64
    %A_row_stride_i64 = "arith.index_cast"(%A_row_stride) : (index) -> i64
    %B_row_stride_i64 = "arith.index_cast"(%B_row_stride) : (index) -> i64
    %D_row_stride_i64 = "arith.index_cast"(%D_row_stride) : (index) -> i64
    %C_row_stride_i64 = "arith.index_cast"(%C_row_stride) : (index) -> i64


    %setup_state = "acc2.setup"(
        %pad_K_shift_or_pad_J_shift_or_pad_I_shift_i64,
        %K_shift_or_J_shift_or_I_shift_i64,
        %A,
        %B,
        %D,
        //%D_selected_i64,
        %C,
        %A_row_stride_i64,
        %B_row_stride_i64,
        %D_row_stride_i64,
        %C_row_stride_i64) 
        <{"accelerator" = "gemmini", "operandSegmentSizes" = array<i32: 10, 0>, 
        "param_names" = [ 
            "k_LOOP_WS_CONFIG_BOUNDS.rs1",
            "k_LOOP_WS_CONFIG_BOUNDS.rs2",
            "k_LOOP_WS_CONFIG_ADDRS_AB.rs1",
            "k_LOOP_WS_CONFIG_ADDRS_AB.rs2",
            "k_LOOP_WS_CONFIG_ADDRS_DC.rs1",
            "k_LOOP_WS_CONFIG_ADDRS_DC.rs2",
            "k_LOOP_WS_CONFIG_STRIDES_AB.rs1",
            "k_LOOP_WS_CONFIG_STRIDES_AB.rs2",
            "k_LOOP_WS_CONFIG_STRIDES_DC.rs1",
            "k_LOOP_WS_CONFIG_STRIDES_DC.rs2"
          ]}> : (i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> !acc2.state<"gemmini">

   // ((uint64_t)(a_spad_id) << 18) | ((uint64_t)(b_spad_id) << 16) | ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate) 
   %a_spad_id_cast = "arith.index_cast"(%a_spad_id) : (i32) -> index
   %b_spad_id_cast = "arith.index_cast"(%b_spad_id) : (i32) -> index
   %act_cast = "arith.index_cast"(%act) : (i32) -> index
   %full_C_cast = "arith.index_cast"(%full_C) : (i1) -> index
   %low_D_cast = "arith.index_cast"(%low_D) : (i1) -> index
   // invert no_bias
   %inv_no_bias = arith.xori %no_bias, %true : i1
   // FIXME: This requires an LLVM cmpi instruction
   // %D_eq_NULL = arith.cmpi eq, %D, %NULLptr : i1
   %ex_accumulate = arith.ori %false, %inv_no_bias : i1 // %D_eq_NULL, %inv_no_bias : i1
   %ex_accumulate_cast = "arith.index_cast"(%ex_accumulate) : (i1) -> index

   %a_spad_id_shift_18 = arith.shli %a_spad_id_cast, %c_18 : index
   %b_spad_id_shift_16 = arith.shli %b_spad_id_cast, %c_16 : index
   %act_shift_8 = arith.shli %act_cast, %c_8 : index
   %low_D_shift_2 = arith.shli %low_D_cast, %c_2 : index
   %full_C_shift_1 = arith.shli %full_C_cast, %c_1 : index

   %or_1 = arith.ori %a_spad_id_shift_18, %b_spad_id_shift_16 : index
   %or_2 = arith.ori %or_1, %act_shift_8 : index
   %or_3 = arith.ori %or_2, %low_D_shift_2 : index
   %or_4 = arith.ori %or_3, %full_C_shift_1 : index
   %or_5 = arith.ori %or_4, %ex_accumulate_cast : index
   %or_5_i64 = "arith.index_cast"(%or_5) : (index) -> i64


   // ((is_resadd) << 2) | ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) 
   %is_resadd_cast = "arith.index_cast"(%false) : (i1) -> index
   %b_transpose_cast = "arith.index_cast"(%b_transpose) : (i1) -> index
   %a_transpose_cast = "arith.index_cast"(%a_transpose) : (i1) -> index
   %is_resadd_shift_2 = arith.shli %is_resadd_cast, %c_2 : index
   %b_transpose_shift_1 = arith.shli %b_transpose_cast , %c_16 : index
   %or_1_1 = arith.ori %is_resadd_shift_2, %b_transpose_cast: index
   %or_1_2 = arith.ori %or_1_1, %a_transpose_cast: index
   %or_1_2_i64 = "arith.index_cast"(%or_1_2) : (index) -> i64


    %token = "acc2.launch"(%or_5_i64, %or_1_2_i64, %setup_state) <{
    "param_names" = ["k_LOOP_WS.rs1", "k_LOOP_WS.rs2"],
    "accelerator" = "gemmini"}> : (i64, i64, !acc2.state<"gemmini">) -> !acc2.token<"gemmini">
      "acc2.await"(%token) : (!acc2.token<"gemmini">) -> ()
    func.return
  }
}


