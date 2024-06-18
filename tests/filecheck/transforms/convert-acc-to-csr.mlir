// RUN: ./compiler/snax-opt %s -p convert-accfg-to-csr --split-input-file | filecheck %s

builtin.module {

  "accfg.accelerator"() <{
      name            = @snax_hwpe_mult,
      fields          = {A=0x3d0, B=0x3d1, O=0x3d3, vector_length=0x3d4, nr_iters=0x3d5, mode=0x3d6},
      launch_fields   = {launch=0x3c0},
      barrier         = 0x3c3
  }> : () -> ()

  func.func public @simple_mult(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xi32>, %i1 : i1) {
    %0 = arith.constant 0 : index
    %cst = arith.constant 0 : i5
    %1 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
    %2 = arith.index_cast %1 : index to i32
    %3 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
    %4 = arith.index_cast %3 : index to i32
    %5 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi32>) -> index
    %6 = arith.index_cast %5 : index to i32
    %7 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index
    %8 = arith.index_cast %7 : index to i32
    %9 = "accfg.setup"(%2, %4, %6, %8) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">
    %10 = "accfg.launch"(%cst, %9) <{"param_names" = ["launch"] ,"accelerator" = "snax_hwpe_mult"}> : (i5, !accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
    "accfg.await"(%10) : (!accfg.token<"snax_hwpe_mult">) -> ()
    %13 = "scf.if"(%i1) ({
      %14 = "accfg.setup"(%6, %9) <{"param_names" = ["B"], "accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 1, 1>}> : (i32, !accfg.state<"snax_hwpe_mult">) -> !accfg.state<"snax_hwpe_mult">
      %15 = "accfg.launch"(%cst, %14) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5, !accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
      "accfg.await"(%15) : (!accfg.token<"snax_hwpe_mult">) -> ()
      %17 = "accfg.setup"(%4, %14) <{"param_names" = ["O"], "accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 1, 1>}> : (i32, !accfg.state<"snax_hwpe_mult">) -> !accfg.state<"snax_hwpe_mult">
      scf.yield %17 : !accfg.state<"snax_hwpe_mult">
    }, {
      %19 = "accfg.launch"(%cst, %9) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5, !accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
      "accfg.await"(%19) : (!accfg.token<"snax_hwpe_mult">) -> ()
      %20 = "accfg.setup"(%6, %4, %9) <{"param_names" = ["B", "O"], "accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 2, 1>}> : (i32, i32, !accfg.state<"snax_hwpe_mult">) -> !accfg.state<"snax_hwpe_mult">
      scf.yield %20 : !accfg.state<"snax_hwpe_mult">
    }) : (i1) -> (!accfg.state<"snax_hwpe_mult">)
    %21 = "accfg.launch"(%cst, %13) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5, !accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
    "accfg.await"(%21) : (!accfg.token<"snax_hwpe_mult">) -> ()
    func.return
  }
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xi32>, %i1 : i1) {
// CHECK-NEXT:     %0 = arith.constant 0 : index
// CHECK-NEXT:     %cst = arith.constant 0 : i5
// CHECK-NEXT:     %1 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
// CHECK-NEXT:     %2 = arith.index_cast %1 : index to i32
// CHECK-NEXT:     %3 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
// CHECK-NEXT:     %4 = arith.index_cast %3 : index to i32
// CHECK-NEXT:     %5 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi32>) -> index
// CHECK-NEXT:     %6 = arith.index_cast %5 : index to i32
// CHECK-NEXT:     %7 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index
// CHECK-NEXT:     %8 = arith.index_cast %7 : index to i32
// CHECK-NEXT:     %9 = arith.constant 976 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%9, %2) <{"asm_string" = "csrw $0, $1", "constraints" = "I, rK", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i32) -> ()
// CHECK-NEXT:     %10 = arith.constant 977 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%10, %4) <{"asm_string" = "csrw $0, $1", "constraints" = "I, rK", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i32) -> ()
// CHECK-NEXT:     %11 = arith.constant 979 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%11, %6) <{"asm_string" = "csrw $0, $1", "constraints" = "I, rK", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i32) -> ()
// CHECK-NEXT:     %12 = arith.constant 981 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%12, %8) <{"asm_string" = "csrw $0, $1", "constraints" = "I, rK", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i32) -> ()
// CHECK-NEXT:     %13 = arith.constant 960 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%13, %cst) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i5) -> ()
// CHECK-NEXT:     scf.while () : () -> () {
// CHECK-NEXT:       %14 = arith.constant 963 : i64
// CHECK-NEXT:       %15 = arith.constant 0 : i32
// CHECK-NEXT:       %16 = "llvm.inline_asm"(%14) <{"asm_string" = "csrr $0, $1", "constraints" = "=r, I", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64) -> i32
// CHECK-NEXT:       %17 = arith.cmpi ne, %16, %15 : i32
// CHECK-NEXT:       scf.condition(%17)
// CHECK-NEXT:     } do {
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %18 = arith.constant 965 : i12
// CHECK-NEXT:     %19 = arith.constant 0 : i5
// CHECK-NEXT:     "llvm.inline_asm"(%18, %19) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i12, i5) -> ()
// CHECK-NEXT:     "llvm.inline_asm"() <{"asm_string" = "nop", "constraints" = "", "asm_dialect" = 0 : i64}> {"has_side_effects"} : () -> ()
// CHECK-NEXT:     "llvm.inline_asm"() <{"asm_string" = "nop", "constraints" = "", "asm_dialect" = 0 : i64}> {"has_side_effects"} : () -> ()
// CHECK-NEXT:     "llvm.inline_asm"() <{"asm_string" = "nop", "constraints" = "", "asm_dialect" = 0 : i64}> {"has_side_effects"} : () -> ()
// CHECK-NEXT:     "scf.if"(%i1) ({
// CHECK-NEXT:       %20 = arith.constant 977 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%20, %6) <{"asm_string" = "csrw $0, $1", "constraints" = "I, rK", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i32) -> ()
// CHECK-NEXT:       %21 = arith.constant 960 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%21, %cst) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i5) -> ()
// CHECK-NEXT:       scf.while () : () -> () {
// CHECK-NEXT:         %22 = arith.constant 963 : i64
// CHECK-NEXT:         %23 = arith.constant 0 : i32
// CHECK-NEXT:         %24 = "llvm.inline_asm"(%22) <{"asm_string" = "csrr $0, $1", "constraints" = "=r, I", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64) -> i32
// CHECK-NEXT:         %25 = arith.cmpi ne, %24, %23 : i32
// CHECK-NEXT:         scf.condition(%25)
// CHECK-NEXT:       } do {
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       %26 = arith.constant 965 : i12
// CHECK-NEXT:       %27 = arith.constant 0 : i5
// CHECK-NEXT:       "llvm.inline_asm"(%26, %27) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i12, i5) -> ()
// CHECK-NEXT:       "llvm.inline_asm"() <{"asm_string" = "nop", "constraints" = "", "asm_dialect" = 0 : i64}> {"has_side_effects"} : () -> ()
// CHECK-NEXT:       "llvm.inline_asm"() <{"asm_string" = "nop", "constraints" = "", "asm_dialect" = 0 : i64}> {"has_side_effects"} : () -> ()
// CHECK-NEXT:       "llvm.inline_asm"() <{"asm_string" = "nop", "constraints" = "", "asm_dialect" = 0 : i64}> {"has_side_effects"} : () -> ()
// CHECK-NEXT:       %28 = arith.constant 979 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%28, %4) <{"asm_string" = "csrw $0, $1", "constraints" = "I, rK", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i32) -> ()
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }, {
// CHECK-NEXT:       %29 = arith.constant 960 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%29, %cst) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i5) -> ()
// CHECK-NEXT:       scf.while () : () -> () {
// CHECK-NEXT:         %30 = arith.constant 963 : i64
// CHECK-NEXT:         %31 = arith.constant 0 : i32
// CHECK-NEXT:         %32 = "llvm.inline_asm"(%30) <{"asm_string" = "csrr $0, $1", "constraints" = "=r, I", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64) -> i32
// CHECK-NEXT:         %33 = arith.cmpi ne, %32, %31 : i32
// CHECK-NEXT:         scf.condition(%33)
// CHECK-NEXT:       } do {
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       %34 = arith.constant 965 : i12
// CHECK-NEXT:       %35 = arith.constant 0 : i5
// CHECK-NEXT:       "llvm.inline_asm"(%34, %35) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i12, i5) -> ()
// CHECK-NEXT:       "llvm.inline_asm"() <{"asm_string" = "nop", "constraints" = "", "asm_dialect" = 0 : i64}> {"has_side_effects"} : () -> ()
// CHECK-NEXT:       "llvm.inline_asm"() <{"asm_string" = "nop", "constraints" = "", "asm_dialect" = 0 : i64}> {"has_side_effects"} : () -> ()
// CHECK-NEXT:       "llvm.inline_asm"() <{"asm_string" = "nop", "constraints" = "", "asm_dialect" = 0 : i64}> {"has_side_effects"} : () -> ()
// CHECK-NEXT:       %36 = arith.constant 977 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%36, %6) <{"asm_string" = "csrw $0, $1", "constraints" = "I, rK", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i32) -> ()
// CHECK-NEXT:       %37 = arith.constant 979 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%37, %4) <{"asm_string" = "csrw $0, $1", "constraints" = "I, rK", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i32) -> ()
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }) : (i1) -> ()
// CHECK-NEXT:     %38 = arith.constant 960 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%38, %cst) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i5) -> ()
// CHECK-NEXT:     scf.while () : () -> () {
// CHECK-NEXT:       %39 = arith.constant 963 : i64
// CHECK-NEXT:       %40 = arith.constant 0 : i32
// CHECK-NEXT:       %41 = "llvm.inline_asm"(%39) <{"asm_string" = "csrr $0, $1", "constraints" = "=r, I", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64) -> i32
// CHECK-NEXT:       %42 = arith.cmpi ne, %41, %40 : i32
// CHECK-NEXT:       scf.condition(%42)
// CHECK-NEXT:     } do {
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %43 = arith.constant 965 : i12
// CHECK-NEXT:     %44 = arith.constant 0 : i5
// CHECK-NEXT:     "llvm.inline_asm"(%43, %44) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i12, i5) -> ()
// CHECK-NEXT:     "llvm.inline_asm"() <{"asm_string" = "nop", "constraints" = "", "asm_dialect" = 0 : i64}> {"has_side_effects"} : () -> ()
// CHECK-NEXT:     "llvm.inline_asm"() <{"asm_string" = "nop", "constraints" = "", "asm_dialect" = 0 : i64}> {"has_side_effects"} : () -> ()
// CHECK-NEXT:     "llvm.inline_asm"() <{"asm_string" = "nop", "constraints" = "", "asm_dialect" = 0 : i64}> {"has_side_effects"} : () -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

builtin.module {

  "accfg.accelerator"() <{
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


  func.func public @test() {
    %t = arith.constant 32 : i32
    %9 = "accfg.setup"(%t,%t,%t,%t,%t,%t,%t,%t,%t,%t) <{"accelerator" = "gemmini", "operandSegmentSizes" = array<i32: 10, 0>, 
    "param_names" = [ "k_LOOP_WS_CONFIG_BOUNDS.rs1",
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs1",
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs1",
        "k_LOOP_WS_CONFIG_STRIDES_AB.rs1",
        "k_LOOP_WS_CONFIG_STRIDES_DC.rs1",
        "k_LOOP_WS_CONFIG_BOUNDS.rs2",
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs2",
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs2",
        "k_LOOP_WS_CONFIG_STRIDES_AB.rs2",
        "k_LOOP_WS_CONFIG_STRIDES_DC.rs2"
        ]}> : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !accfg.state<"gemmini">

    %10 = "accfg.launch"(%t,%t,%9) <{
    "param_names" = ["k_LOOP_WS.rs1", "k_LOOP_WS.rs2"],
    "accelerator" = "gemmini"}> : (i32, i32, !accfg.state<"gemmini">) -> !accfg.token<"gemmini">
      "accfg.await"(%10) : (!accfg.token<"gemmini">) -> ()
    func.return
  }
}

// CHECK: builtin.module {
// CHECK-NEXT:   func.func public @test() {
// CHECK-NEXT:     %t = arith.constant 32 : i32
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{"asm_string" = ".insn r CUSTOM_3, 0x3, 9 ,x0, $0, $1", "constraints" = "r, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i32, i32) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{"asm_string" = ".insn r CUSTOM_3, 0x3, 10 ,x0, $0, $1", "constraints" = "r, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i32, i32) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{"asm_string" = ".insn r CUSTOM_3, 0x3, 11 ,x0, $0, $1", "constraints" = "r, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i32, i32) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{"asm_string" = ".insn r CUSTOM_3, 0x3, 12 ,x0, $0, $1", "constraints" = "r, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i32, i32) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{"asm_string" = ".insn r CUSTOM_3, 0x3, 13 ,x0, $0, $1", "constraints" = "r, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i32, i32) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{"asm_string" = ".insn r CUSTOM_3, 0x3, 8 ,x0, $0, $1", "constraints" = "r, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i32, i32) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
