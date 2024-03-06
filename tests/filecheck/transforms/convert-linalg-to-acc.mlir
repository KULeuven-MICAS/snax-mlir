// XFAIL: *
// RUN: ./compiler/snax-opt -p convert-linalg-to-acc,mlir-opt[cse,canonicalize] %s | filecheck %s

"builtin.module"() ({
  func.func public @simple_mult(
    %A: memref<?xi32>,
    %B: memref<?xi32>,
    %D: memref<?xi32>
  ) -> () {
    linalg.generic { indexing_maps = [], iterator_types = ["parallel"], library_call = "snax_hwpe_mult" }
    ins(%A, %B: memref<?xi32>, memref<?xi32>)
      outs(%D: memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %d: i32):
      %r0 = arith.muli %a, %b : i32
      linalg.yield %r0 : i32
    }

    %i1 = "test.op"() : () -> i1

    %v_final = "scf.if"(%i1) ({

      linalg.generic { indexing_maps = [], iterator_types = ["parallel"], library_call = "snax_hwpe_mult" }
      ins(%A, %D: memref<?xi32>, memref<?xi32>) outs(%D: memref<?xi32>) {
      ^bb0(%a: i32, %b: i32, %d: i32):
          %r0 = arith.muli %a, %b : i32
          linalg.yield %r0 : i32
      }
      %v1 = "test.op"() : () -> i32

      scf.yield %v1 : i32
    }, {
      %v2 = "test.op"() : () -> i32

      linalg.generic { indexing_maps = [], iterator_types = ["parallel"], library_call = "snax_hwpe_mult" }
      ins(%A, %B: memref<?xi32>, memref<?xi32>) outs(%D: memref<?xi32>) {
      ^bb0(%a: i32, %b: i32, %d: i32):
          %r0 = arith.muli %a, %b : i32
          linalg.yield %r0 : i32
      }

      scf.yield %v2 : i32
    }) : (i1) -> i32

    "test.op"(%v_final) : (i32) -> ()

    linalg.generic { indexing_maps = [], iterator_types = ["parallel"], library_call = "snax_hwpe_mult" }
    ins(%A, %D: memref<?xi32>, memref<?xi32>) outs(%B: memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %d: i32):
      %r0 = arith.muli %a, %b : i32
      linalg.yield %r0 : i32
    }

    func.return
  }
}): () -> ()


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xi32>) {
// CHECK-NEXT:     %0 = arith.constant 0 : index
// CHECK-NEXT:     %1 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
// CHECK-NEXT:     %2 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
// CHECK-NEXT:     %3 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi32>) -> index
// CHECK-NEXT:     %4 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index
// CHECK-NEXT:     %5 = "acc2.setup"(%1, %2, %3, %4) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index) -> !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     %6 = "acc2.launch"(%5) <{"accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token
// CHECK-NEXT:     "acc2.await"(%6) : (!acc2.token) -> ()
// CHECK-NEXT:     %7 = "test.op"() : () -> i1
// CHECK-NEXT:     %8, %9 = "scf.if"(%7) ({
// CHECK-NEXT:       %10 = "acc2.setup"(%1, %3, %3, %4, %5) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 1>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:       %11 = "acc2.launch"(%10) <{"accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token
// CHECK-NEXT:       "acc2.await"(%11) : (!acc2.token) -> ()
// CHECK-NEXT:       %12 = "test.op"() : () -> i32
// CHECK-NEXT:       scf.yield %12, %10 : i32, !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     }, {
// CHECK-NEXT:       %13 = "test.op"() : () -> i32
// CHECK-NEXT:       %14 = "acc2.setup"(%1, %2, %3, %4, %5) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 1>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:       %15 = "acc2.launch"(%14) <{"accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token
// CHECK-NEXT:       "acc2.await"(%15) : (!acc2.token) -> ()
// CHECK-NEXT:       scf.yield %13, %14 : i32, !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     }) : (i1) -> (i32, !acc2.state<"snax_hwpe_mult">)
// CHECK-NEXT:     "test.op"(%8) : (i32) -> ()
// CHECK-NEXT:     %16 = "acc2.setup"(%1, %3, %2, %4, %9) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 1>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     %17 = "acc2.launch"(%16) <{"accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token
// CHECK-NEXT:     "acc2.await"(%17) : (!acc2.token) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   "acc2.accelerator"() <{"name" = @snax_hwpe_mult, "fields" = {"A" = 960 : i32, "B" = 961 : i32, "O" = 962 : i32, "size" = 963 : i32}, "launch_addr" = 975 : i32, "barrier_enable" = 1987 : i32, "barrier_trigger" = 1988 : i32}> : () -> ()
// CHECK-NEXT: }
