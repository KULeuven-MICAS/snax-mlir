// RUN: ./compiler/snax-opt %s -p convert-acc-to-csr | filecheck %s

builtin.module {

  "acc2.accelerator"() <{
      name            = @snax_hwpe_mult,
      fields          = {A=0x3d0, B=0x3d1, O=0x3d3, nr_iters=0x3d4, vector_length=0x3d5, mode=0x3d6},
      launch_addr     = 0x3c0,
      barrier_sw_barrier  = 0x3c3
  }> : () -> ()

  func.func public @simple_mult(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xi32>, %i1 : i1) {
    %0 = arith.constant 0 : index
    %1 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
    %2 = "arith.index_cast"(%1) : (index) -> i32
    %3 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
    %4 = "arith.index_cast"(%3) : (index) -> i32
    %5 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi32>) -> index
    %6 = "arith.index_cast"(%5) : (index) -> i32
    %7 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index
    %8 = "arith.index_cast"(%7) : (index) -> i32
    %9 = "acc2.setup"(%2, %4, %6, %8) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">
    %10 = "acc2.launch"(%9) <{"accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
    "acc2.await"(%10) : (!acc2.token<"snax_hwpe_mult">) -> ()
    %13 = "scf.if"(%i1) ({
      %14 = "acc2.setup"(%6, %9) <{"param_names" = ["B"], "accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 1, 1>}> : (i32, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">
      %15 = "acc2.launch"(%14) <{"accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
      "acc2.await"(%15) : (!acc2.token<"snax_hwpe_mult">) -> ()
      %17 = "acc2.setup"(%4, %14) <{"param_names" = ["O"], "accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 1, 1>}> : (i32, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">
      scf.yield %17 : !acc2.state<"snax_hwpe_mult">
    }, {
      %19 = "acc2.launch"(%9) <{"accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
      "acc2.await"(%19) : (!acc2.token<"snax_hwpe_mult">) -> ()
      %20 = "acc2.setup"(%6, %4, %9) <{"param_names" = ["B", "O"], "accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 2, 1>}> : (i32, i32, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">
      scf.yield %20 : !acc2.state<"snax_hwpe_mult">
    }) : (i1) -> (!acc2.state<"snax_hwpe_mult">)
    %21 = "acc2.launch"(%13) <{"accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
    "acc2.await"(%21) : (!acc2.token<"snax_hwpe_mult">) -> ()
    func.return
  }
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xi32>, %i1 : i1) {
// CHECK-NEXT:     %0 = arith.constant 0 : index
// CHECK-NEXT:     %1 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
// CHECK-NEXT:     %2 = "arith.index_cast"(%1) : (index) -> i32
// CHECK-NEXT:     %3 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
// CHECK-NEXT:     %4 = "arith.index_cast"(%3) : (index) -> i32
// CHECK-NEXT:     %5 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi32>) -> index
// CHECK-NEXT:     %6 = "arith.index_cast"(%5) : (index) -> i32
// CHECK-NEXT:     %7 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index
// CHECK-NEXT:     %8 = "arith.index_cast"(%7) : (index) -> i32
// CHECK-NEXT:     %9 = arith.constant 976 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%9, %2) <{"asm_string" = "csrw $0, $1", "constraints" = "I, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i32) -> ()
// CHECK-NEXT:     %10 = arith.constant 977 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%10, %4) <{"asm_string" = "csrw $0, $1", "constraints" = "I, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i32) -> ()
// CHECK-NEXT:     %11 = arith.constant 979 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%11, %6) <{"asm_string" = "csrw $0, $1", "constraints" = "I, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i32) -> ()
// CHECK-NEXT:     %12 = arith.constant 980 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%12, %8) <{"asm_string" = "csrw $0, $1", "constraints" = "I, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i32) -> ()
// CHECK-NEXT:     %13 = arith.constant 960 : i64
// CHECK-NEXT:     %14 = arith.constant 1 : i5
// CHECK-NEXT:     "llvm.inline_asm"(%13, %14) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i5) -> ()
// CHECK-NEXT:     %15 = arith.constant 963 : i64
// CHECK-NEXT:     %16 = arith.constant 0 : i5
// CHECK-NEXT:     "llvm.inline_asm"(%15, %16) <{"asm_string" = "\n  csrr a0, $0\n1:\n  bnez a0, 1b\n  csrwi 0x3c5, $1\n  nop\n  nop\n  nop\n", "constraints" = "I,K", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i5) -> ()
// CHECK-NEXT:     "scf.if"(%i1) ({
// CHECK-NEXT:       %17 = arith.constant 977 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%17, %6) <{"asm_string" = "csrw $0, $1", "constraints" = "I, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i32) -> ()
// CHECK-NEXT:       %18 = arith.constant 960 : i64
// CHECK-NEXT:       %19 = arith.constant 1 : i5
// CHECK-NEXT:       "llvm.inline_asm"(%18, %19) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i5) -> ()
// CHECK-NEXT:       %20 = arith.constant 963 : i64
// CHECK-NEXT:       %21 = arith.constant 0 : i5
// CHECK-NEXT:       "llvm.inline_asm"(%20, %21) <{"asm_string" = "\n  csrr a0, $0\n1:\n  bnez a0, 1b\n  csrwi 0x3c5, $1\n  nop\n  nop\n  nop\n", "constraints" = "I,K", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i5) -> ()
// CHECK-NEXT:       %22 = arith.constant 979 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%22, %4) <{"asm_string" = "csrw $0, $1", "constraints" = "I, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i32) -> ()
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }, {
// CHECK-NEXT:       %23 = arith.constant 960 : i64
// CHECK-NEXT:       %24 = arith.constant 1 : i5
// CHECK-NEXT:       "llvm.inline_asm"(%23, %24) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i5) -> ()
// CHECK-NEXT:       %25 = arith.constant 963 : i64
// CHECK-NEXT:       %26 = arith.constant 0 : i5
// CHECK-NEXT:       "llvm.inline_asm"(%25, %26) <{"asm_string" = "\n  csrr a0, $0\n1:\n  bnez a0, 1b\n  csrwi 0x3c5, $1\n  nop\n  nop\n  nop\n", "constraints" = "I,K", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i5) -> ()
// CHECK-NEXT:       %27 = arith.constant 977 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%27, %6) <{"asm_string" = "csrw $0, $1", "constraints" = "I, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i32) -> ()
// CHECK-NEXT:       %28 = arith.constant 979 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%28, %4) <{"asm_string" = "csrw $0, $1", "constraints" = "I, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i32) -> ()
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }) : (i1) -> ()
// CHECK-NEXT:     %29 = arith.constant 960 : i64
// CHECK-NEXT:     %30 = arith.constant 1 : i5
// CHECK-NEXT:     "llvm.inline_asm"(%29, %30) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i5) -> ()
// CHECK-NEXT:     %31 = arith.constant 963 : i64
// CHECK-NEXT:     %32 = arith.constant 0 : i5
// CHECK-NEXT:     "llvm.inline_asm"(%31, %32) <{"asm_string" = "\n  csrr a0, $0\n1:\n  bnez a0, 1b\n  csrwi 0x3c5, $1\n  nop\n  nop\n  nop\n", "constraints" = "I,K", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i64, i5) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }


