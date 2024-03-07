// RUN: ./compiler/snax-opt %s -p convert-acc-to-csr | filecheck %s

builtin.module {

  "acc2.accelerator"() <{
      name            = @snax_hwpe_mult,
      fields          = {A=0x3c0, B=0x3c1, O=0x3c2, size=0x3c3},
      launch_addr     = 0x3cf,
      barrier_enable  = 0x7c3,
      barrier_trigger = 0x7c4
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
    %9 = "acc2.setup"(%2, %4, %6, %8) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "size"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">
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
// CHECK-NEXT:     %9 = arith.constant 960 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%9, %2) <{"asm_string" = "csrw $0, $1", "constraints" = "I, r", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i32) -> ()
// CHECK-NEXT:     %10 = arith.constant 961 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%10, %4) <{"asm_string" = "csrw $0, $1", "constraints" = "I, r", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i32) -> ()
// CHECK-NEXT:     %11 = arith.constant 962 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%11, %6) <{"asm_string" = "csrw $0, $1", "constraints" = "I, r", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i32) -> ()
// CHECK-NEXT:     %12 = arith.constant 963 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%12, %8) <{"asm_string" = "csrw $0, $1", "constraints" = "I, r", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i32) -> ()
// CHECK-NEXT:     %13 = arith.constant 975 : i64
// CHECK-NEXT:     %14 = arith.constant 1 : i5
// CHECK-NEXT:     "llvm.inline_asm"(%13, %14) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i5) -> ()
// CHECK-NEXT:     %15 = arith.constant 1987 : i64
// CHECK-NEXT:     %16 = arith.constant 1988 : i64
// CHECK-NEXT:     %17 = arith.constant 1 : i5
// CHECK-NEXT:     %18 = arith.constant 0 : i5
// CHECK-NEXT:     "llvm.inline_asm"(%15, %17) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i5) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%16, %18) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i5) -> ()
// CHECK-NEXT:     "scf.if"(%i1) ({
// CHECK-NEXT:       %19 = arith.constant 961 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%19, %6) <{"asm_string" = "csrw $0, $1", "constraints" = "I, r", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i32) -> ()
// CHECK-NEXT:       %20 = arith.constant 975 : i64
// CHECK-NEXT:       %21 = arith.constant 1 : i5
// CHECK-NEXT:       "llvm.inline_asm"(%20, %21) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i5) -> ()
// CHECK-NEXT:       %22 = arith.constant 1987 : i64
// CHECK-NEXT:       %23 = arith.constant 1988 : i64
// CHECK-NEXT:       %24 = arith.constant 1 : i5
// CHECK-NEXT:       %25 = arith.constant 0 : i5
// CHECK-NEXT:       "llvm.inline_asm"(%22, %24) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i5) -> ()
// CHECK-NEXT:       "llvm.inline_asm"(%23, %25) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i5) -> ()
// CHECK-NEXT:       %26 = arith.constant 962 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%26, %4) <{"asm_string" = "csrw $0, $1", "constraints" = "I, r", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i32) -> ()
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }, {
// CHECK-NEXT:       %27 = arith.constant 975 : i64
// CHECK-NEXT:       %28 = arith.constant 1 : i5
// CHECK-NEXT:       "llvm.inline_asm"(%27, %28) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i5) -> ()
// CHECK-NEXT:       %29 = arith.constant 1987 : i64
// CHECK-NEXT:       %30 = arith.constant 1988 : i64
// CHECK-NEXT:       %31 = arith.constant 1 : i5
// CHECK-NEXT:       %32 = arith.constant 0 : i5
// CHECK-NEXT:       "llvm.inline_asm"(%29, %31) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i5) -> ()
// CHECK-NEXT:       "llvm.inline_asm"(%30, %32) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i5) -> ()
// CHECK-NEXT:       %33 = arith.constant 961 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%33, %6) <{"asm_string" = "csrw $0, $1", "constraints" = "I, r", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i32) -> ()
// CHECK-NEXT:       %34 = arith.constant 962 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%34, %4) <{"asm_string" = "csrw $0, $1", "constraints" = "I, r", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i32) -> ()
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }) : (i1) -> ()
// CHECK-NEXT:     %35 = arith.constant 975 : i64
// CHECK-NEXT:     %36 = arith.constant 1 : i5
// CHECK-NEXT:     "llvm.inline_asm"(%35, %36) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i5) -> ()
// CHECK-NEXT:     %37 = arith.constant 1987 : i64
// CHECK-NEXT:     %38 = arith.constant 1988 : i64
// CHECK-NEXT:     %39 = arith.constant 1 : i5
// CHECK-NEXT:     %40 = arith.constant 0 : i5
// CHECK-NEXT:     "llvm.inline_asm"(%37, %39) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i5) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%38, %40) <{"asm_string" = "csrw $0, $1", "constraints" = "I, K", "asm_dialect" = 0 : i64, "has_side_effects"}> : (i64, i5) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
