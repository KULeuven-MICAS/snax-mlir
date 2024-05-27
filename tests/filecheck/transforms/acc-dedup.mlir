// RUN: ./compiler/snax-opt %s -p acc-dedup | filecheck %s

func.func public @simple_mult(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xi32>) {
  %0 = arith.constant 0 : index
  %cst_0 = arith.constant 0 : i5
  %1 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
  %2 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
  %3 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi32>) -> index
  %4 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index

  %5 = "acc2.setup"(%1, %2, %3, %4) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index) -> !acc2.state<"snax_hwpe_mult">

  %6 = "acc2.launch"(%cst_0, %5) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5,!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
  "acc2.await"(%6) : (!acc2.token<"snax_hwpe_mult">) -> ()

  %7 = "test.op"() : () -> i1

  %8, %9 = "scf.if"(%7) ({
    %10 = "acc2.setup"(%1, %3, %3, %4, %5) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 1>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">

    %11 = "acc2.launch"(%cst_0, %10) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5,!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
    "acc2.await"(%11) : (!acc2.token<"snax_hwpe_mult">) -> ()

    %12 = "test.op"() : () -> i32
    scf.yield %12, %10 : i32, !acc2.state<"snax_hwpe_mult">
  }, {
    %13 = "test.op"() : () -> i32

    %14 = "acc2.setup"(%1, %2, %3, %4, %5) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 1>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">
    %15 = "acc2.launch"(%cst_0, %14) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5,!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
    "acc2.await"(%15) : (!acc2.token<"snax_hwpe_mult">) -> ()

    scf.yield %13, %14 : i32, !acc2.state<"snax_hwpe_mult">
  }) : (i1) -> (i32, !acc2.state<"snax_hwpe_mult">)

  "test.op"(%8) : (i32) -> ()

  %16 = "acc2.setup"(%1, %3, %2, %4, %9) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 1>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">
  %17 = "acc2.launch"(%cst_0, %16) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5,!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
  "acc2.await"(%17) : (!acc2.token<"snax_hwpe_mult">) -> ()


  %lb = arith.constant 0 : index
  %ub = arith.constant 100 : index
  %step = arith.constant 100 : index

  %res_1 = scf.for %iv = %lb to %ub step %step iter_args(%inner_state = %16) -> (!acc2.state<"snax_hwpe_mult">) {

    %s_new = "acc2.setup"(%1, %2, %3, %iv, %inner_state) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 1>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">
    %222 = "acc2.launch"(%cst_0, %s_new) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5,!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
    "acc2.await"(%222) : (!acc2.token<"snax_hwpe_mult">) -> ()

    scf.yield %s_new : !acc2.state<"snax_hwpe_mult">
  }

  func.return
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xi32>) {
// CHECK-NEXT:     %0 = arith.constant 0 : index
// CHECK-NEXT:     %cst_0 = arith.constant 0 : i5
// CHECK-NEXT:     %1 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
// CHECK-NEXT:     %2 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
// CHECK-NEXT:     %3 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi32>) -> index
// CHECK-NEXT:     %4 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index
// CHECK-NEXT:     %5 = "acc2.setup"(%1, %2, %3, %4) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index) -> !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     %6 = "acc2.launch"(%cst_0, %5) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5, !acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
// CHECK-NEXT:     "acc2.await"(%6) : (!acc2.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:     %7 = "test.op"() : () -> i1
// CHECK-NEXT:     %8, %9 = "scf.if"(%7) ({
// CHECK-NEXT:       %10 = "acc2.setup"(%3, %5) <{"param_names" = ["B"], "accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 1, 1>}> : (index, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:       %11 = "acc2.launch"(%cst_0, %10) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5, !acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
// CHECK-NEXT:       "acc2.await"(%11) : (!acc2.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:       %12 = "test.op"() : () -> i32
// CHECK-NEXT:       %13 = "acc2.setup"(%2, %10) <{"param_names" = ["O"], "accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 1, 1>}> : (index, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:       scf.yield %12, %13 : i32, !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     }, {
// CHECK-NEXT:       %14 = "test.op"() : () -> i32
// CHECK-NEXT:       %15 = "acc2.launch"(%cst_0, %5) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5, !acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
// CHECK-NEXT:       "acc2.await"(%15) : (!acc2.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:       %16 = "acc2.setup"(%3, %2, %5) <{"param_names" = ["B", "O"], "accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 2, 1>}> : (index, index, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:       scf.yield %14, %16 : i32, !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     }) : (i1) -> (i32, !acc2.state<"snax_hwpe_mult">)
// CHECK-NEXT:     "test.op"(%8) : (i32) -> ()
// CHECK-NEXT:     %17 = "acc2.launch"(%cst_0, %9) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5, !acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
// CHECK-NEXT:     "acc2.await"(%17) : (!acc2.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
