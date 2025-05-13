// RUN: snax-opt --split-input-file -p 'convert-linalg-to-accfg,mlir-opt{executable=mlir-opt generic=true arguments=-cse,-canonicalize,-allow-unregistered-dialect,-mlir-print-op-generic,-split-input-file},accfg-dedup,convert-accfg-to-csr' %s | filecheck %s

builtin.module {
  "accfg.accelerator"() <{
      name            = @snax_hwpe_mult,
      fields          = {A=0x3d0, B=0x3d1, O=0x3d3, vector_length=0x3d4, nr_iters=0x3d5, mode=0x3d6},
      launch_fields   = {launch=0x3c0},
      barrier         = 0x3c3
  }> : () -> ()

  func.func @scf_for(%A: i32, %B: i32, %O: i32) {
    %c32 = arith.constant 32 : i32

    // set up the loop
    %lb = arith.constant 0 : index
    %ub = arith.constant 100 : index
    %step = arith.constant 1 : index

    scf.for %iv = %lb to %ub step %step {
      // some variables computed in-loop:
      %B_shift = arith.addi %B, %c32 : i32
      %O_shift = arith.addi %O, %c32 : i32

      // launch with loop-invariant and loop-dependent vars:
      %state = accfg.setup "snax_hwpe_mult" to ("A" = %A : i32, "B" = %B_shift : i32, "O" = %O_shift : i32, "vector_length" = %c32 : i32) : !accfg.state<"snax_hwpe_mult">
      %token = "accfg.launch"(%c32, %state) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i32, !accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
      "accfg.await"(%token) : (!accfg.token<"snax_hwpe_mult">) -> ()

      scf.yield
    }

    return
  }

}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func @scf_for(%arg0 : i32, %arg1 : i32, %arg2 : i32) {
// CHECK-NEXT:     %0 = arith.constant 32 : i32
// CHECK-NEXT:     %1 = arith.constant 0 : index
// CHECK-NEXT:     %2 = arith.constant 100 : index
// CHECK-NEXT:     %3 = arith.constant 1 : index
// CHECK-NEXT:     %4 = arith.constant 976 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%4, %arg0) <{asm_string = "csrw $0, $1", constraints = "I, rK", asm_dialect = 0 : i64, has_side_effects}> : (i64, i32) -> ()
// CHECK-NEXT:     %5 = arith.constant 980 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%5, %0) <{asm_string = "csrw $0, $1", constraints = "I, rK", asm_dialect = 0 : i64, has_side_effects}> : (i64, i32) -> ()
// CHECK-NEXT:     scf.for %arg3 = %1 to %2 step %3 {
// CHECK-NEXT:       %6 = arith.addi %arg1, %0 : i32
// CHECK-NEXT:       %7 = arith.addi %arg2, %0 : i32
// CHECK-NEXT:       %8 = arith.constant 977 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%8, %6) <{asm_string = "csrw $0, $1", constraints = "I, rK", asm_dialect = 0 : i64, has_side_effects}> : (i64, i32) -> ()
// CHECK-NEXT:       %9 = arith.constant 979 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%9, %7) <{asm_string = "csrw $0, $1", constraints = "I, rK", asm_dialect = 0 : i64, has_side_effects}> : (i64, i32) -> ()
// CHECK-NEXT:       %10 = arith.constant 960 : i64
// CHECK-NEXT:       "llvm.inline_asm"(%10, %0) <{asm_string = "csrw $0, $1", constraints = "I, K", asm_dialect = 0 : i64, has_side_effects}> : (i64, i32) -> ()
// CHECK-NEXT:       scf.while () : () -> () {
// CHECK-NEXT:         %11 = arith.constant 963 : i64
// CHECK-NEXT:         %12 = arith.constant 0 : i32
// CHECK-NEXT:         %13 = "llvm.inline_asm"(%11) <{asm_string = "csrr $0, $1", constraints = "=r, I", asm_dialect = 0 : i64, has_side_effects}> : (i64) -> i32
// CHECK-NEXT:         %14 = arith.cmpi ne, %13, %12 : i32
// CHECK-NEXT:         scf.condition(%14)
// CHECK-NEXT:       } do {
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       %15 = arith.constant 965 : i12
// CHECK-NEXT:       %16 = arith.constant 0 : i5
// CHECK-NEXT:       "llvm.inline_asm"(%15, %16) <{asm_string = "csrw $0, $1", constraints = "I, K", asm_dialect = 0 : i64, has_side_effects}> : (i12, i5) -> ()
// Check any amount of nops
// CHECK-NEXT:       "llvm.inline_asm"() <{asm_string = "nop", constraints = "", asm_dialect = 0 : i64, has_side_effects}> : () -> ()
// CHECK: }
// CHECK:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
