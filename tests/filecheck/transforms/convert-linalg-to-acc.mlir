// RUN: ./compiler/snax-opt -p convert-linalg-to-acc,mlir-opt{executable=mlir-opt-17\ generic=true\ arguments='-cse,-canonicalize,-allow-unregistered-dialect,-mlir-print-op-generic'} %s | filecheck %s

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
// CHECK-NEXT:     %1 = arith.constant 1 : i32
// CHECK-NEXT:     %2 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
// CHECK-NEXT:     %3 = "arith.index_cast"(%2) : (index) -> i32
// CHECK-NEXT:     %4 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
// CHECK-NEXT:     %5 = "arith.index_cast"(%4) : (index) -> i32
// CHECK-NEXT:     %6 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi32>) -> index
// CHECK-NEXT:     %7 = "arith.index_cast"(%6) : (index) -> i32
// CHECK-NEXT:     %8 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index
// CHECK-NEXT:     %9 = "arith.index_cast"(%8) : (index) -> i32
// CHECK-NEXT:     %10 = "acc2.setup"(%3, %5, %7, %1, %9, %1) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 6, 0>, "param_names" = ["A", "B", "O", "vector_length", "nr_iters", "mode"]}> : (i32, i32, i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     %11 = "acc2.launch"(%10) <{"accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
// CHECK-NEXT:     "acc2.await"(%11) : (!acc2.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:     %12 = "test.op"() : () -> i1
// CHECK-NEXT:     %13, %14 = "scf.if"(%12) ({
// CHECK-NEXT:       %15 = "acc2.setup"(%3, %7, %7, %1, %9, %1, %10) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 6, 1>, "param_names" = ["A", "B", "O", "vector_length", "nr_iters", "mode"]}> : (i32, i32, i32, i32, i32, i32, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:       %16 = "acc2.launch"(%15) <{"accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
// CHECK-NEXT:       "acc2.await"(%16) : (!acc2.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:       %17 = "test.op"() : () -> i32
// CHECK-NEXT:       scf.yield %17, %15 : i32, !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     }, {
// CHECK-NEXT:       %18 = "test.op"() : () -> i32
// CHECK-NEXT:       %19 = "acc2.setup"(%3, %5, %7, %1, %9, %1, %10) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 6, 1>, "param_names" = ["A", "B", "O", "vector_length", "nr_iters", "mode"]}> : (i32, i32, i32, i32, i32, i32, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:       %20 = "acc2.launch"(%19) <{"accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
// CHECK-NEXT:       "acc2.await"(%20) : (!acc2.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:       scf.yield %18, %19 : i32, !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     }) : (i1) -> (i32, !acc2.state<"snax_hwpe_mult">)
// CHECK-NEXT:     "test.op"(%13) : (i32) -> ()
// CHECK-NEXT:     %21 = "acc2.setup"(%3, %7, %5, %1, %9, %1, %14) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 6, 1>, "param_names" = ["A", "B", "O", "vector_length", "nr_iters", "mode"]}> : (i32, i32, i32, i32, i32, i32, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     %22 = "acc2.launch"(%21) <{"accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
// CHECK-NEXT:     "acc2.await"(%22) : (!acc2.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   "acc2.accelerator"() <{"barrier" = 963 : i32, "fields" = {"A" = 976 : i32, "B" = 977 : i32, "O" = 979 : i32, "mode" = 982 : i32, "nr_iters" = 981 : i32, "vector_length" = 980 : i32}, "launch_addr" = 960 : i32, "name" = @snax_hwpe_mult}> : () -> ()
// CHECK-NEXT: } 
