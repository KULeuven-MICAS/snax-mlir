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
  %step = arith.constant 1 : index

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
// CHECK-NEXT:     %5 = acc2.setup on "snax_hwpe_mult" ("A" = %1 : index, "B" = %2 : index, "O" = %3 : index, "size" = %4 : index) : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     %6 = "acc2.launch"(%cst_0, %5) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5, !acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
// CHECK-NEXT:     "acc2.await"(%6) : (!acc2.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:     %7 = "test.op"() : () -> i1
// CHECK-NEXT:     %8, %9 = "scf.if"(%7) ({
// CHECK-NEXT:       %10 = acc2.setup on "snax_hwpe_mult" ("B" = %3 : index) in_state(%5) : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:       %11 = "acc2.launch"(%cst_0, %10) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5, !acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
// CHECK-NEXT:       "acc2.await"(%11) : (!acc2.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:       %12 = "test.op"() : () -> i32
// CHECK-NEXT:       %13 = acc2.setup on "snax_hwpe_mult" ("O" = %2 : index) in_state(%10) : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:       scf.yield %12, %13 : i32, !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     }, {
// CHECK-NEXT:       %14 = "test.op"() : () -> i32
// CHECK-NEXT:       %15 = "acc2.launch"(%cst_0, %5) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5, !acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
// CHECK-NEXT:       "acc2.await"(%15) : (!acc2.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:       %16 = acc2.setup on "snax_hwpe_mult" ("B" = %3 : index, "O" = %2 : index) in_state(%5) : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:       scf.yield %14, %16 : i32, !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     }) : (i1) -> (i32, !acc2.state<"snax_hwpe_mult">)
// CHECK-NEXT:     "test.op"(%8) : (i32) -> ()
// CHECK-NEXT:     %17 = "acc2.launch"(%cst_0, %9) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5, !acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
// CHECK-NEXT:     "acc2.await"(%17) : (!acc2.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:     %lb = arith.constant 0 : index
// CHECK-NEXT:     %ub = arith.constant 100 : index
// CHECK-NEXT:     %step = arith.constant 1 : index
// CHECK-NEXT:     %18 = acc2.setup on "snax_hwpe_mult" ("B" = %2 : index, "O" = %3 : index) in_state(%9) : !acc2.state<"snax_hwpe_mult">
//                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ new setup op for invariant ops
// CHECK-NEXT:     %res_1 = scf.for %iv = %lb to %ub step %step iter_args(%inner_state = %18) -> (!acc2.state<"snax_hwpe_mult">) {
// CHECK-NEXT:        %s_new = acc2.setup on "snax_hwpe_mult" ("size" = %iv : index) in_state(%inner_state) : !acc2.state<"snax_hwpe_mult">
//                                   only loop-dependent vars remaining ^^^
// CHECK-NEXT:       %19 = "acc2.launch"(%cst_0, %s_new) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5, !acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
// CHECK-NEXT:       "acc2.await"(%19) : (!acc2.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:       scf.yield %s_new : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }


func.func @scf_for_test(%A: i32, %B: i32) {
  %O = "test.op"() : () -> i32
  %c32 = arith.constant 32 : i32

  // initial launch
  %init = acc2.setup on "snax_hwpe_mult" ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "size" = %c32 : i32) : !acc2.state<"snax_hwpe_mult">
  %token = "acc2.launch"(%init) <{"param_names" = [], "accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
  "acc2.await"(%token) : (!acc2.token<"snax_hwpe_mult">) -> ()

  // set up the loop
  %lb = arith.constant 0 : index
  %ub = arith.constant 100 : index
  %step = arith.constant 1 : index

  // set up one loop-invariant operand
  %A_shift = arith.addi %A, %c32 : i32

  // iterate a bunch of times
  %res_state = scf.for %iv = %lb to %ub step %step iter_args(%inner_state = %init) -> (!acc2.state<"snax_hwpe_mult">) {
    // some variables computed in-loop:
    %B_shift = arith.addi %B, %c32 : i32
    %O_shift = arith.addi %O, %c32 : i32

    // launch with loop-invariant and loop-dependent vars:
    %s_new = acc2.setup on "snax_hwpe_mult" ("A" = %A_shift : i32, "B" = %B_shift : i32, "O" = %O_shift : i32, "size" = %c32 : i32) in_state(%inner_state) : !acc2.state<"snax_hwpe_mult">
    %tok = "acc2.launch"(%s_new) <{"param_names" = [], "accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
    "acc2.await"(%tok) : (!acc2.token<"snax_hwpe_mult">) -> ()

    scf.yield %s_new : !acc2.state<"snax_hwpe_mult">
  }

  // tailing launch, with same inputs as initial launch:
  %final = acc2.setup on "snax_hwpe_mult" ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "size" = %c32 : i32) in_state(%res_state) : !acc2.state<"snax_hwpe_mult">
  %token2 = "acc2.launch"(%final) <{"param_names" = [], "accelerator" = "snax_hwpe_mult"}> :  (!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
  "acc2.await"(%token2) : (!acc2.token<"snax_hwpe_mult">) -> ()
  return
}

// CHECK-NEXT:   func.func @scf_for_test(%A : i32, %B : i32) {
// CHECK-NEXT:     %O = "test.op"() : () -> i32
// CHECK-NEXT:     %c32 = arith.constant 32 : i32
// CHECK-NEXT:     %init = acc2.setup on "snax_hwpe_mult" ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "size" = %c32 : i32) : !acc2.state<"snax_hwpe_mult">
//                 ^^^ first setup should be untouched ^^^
// CHECK-NEXT:     %token = "acc2.launch"(%init) <{"param_names" = [], "accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
// CHECK-NEXT:     "acc2.await"(%token) : (!acc2.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:     %lb_1 = arith.constant 0 : index
// CHECK-NEXT:     %ub_1 = arith.constant 100 : index
// CHECK-NEXT:     %step_1 = arith.constant 1 : index
// CHECK-NEXT:     %A_shift = arith.addi %A, %c32 : i32
// CHECK-NEXT:     %20 = acc2.setup on "snax_hwpe_mult" ("A" = %A_shift : i32) in_state(%init) : !acc2.state<"snax_hwpe_mult">
//                       ^^^^^^^^^^^^^^^^^^^^^ new setup op for loop-invariant variables
// CHECK-NEXT:     %res_state = scf.for %iv_1 = %lb_1 to %ub_1 step %step_1 iter_args(%inner_state_1 = %20) -> (!acc2.state<"snax_hwpe_mult">) {
// CHECK-NEXT:       %B_shift = arith.addi %B, %c32 : i32
// CHECK-NEXT:       %O_shift = arith.addi %O, %c32 : i32
// CHECK-NEXT:       %s_new_1 = acc2.setup on "snax_hwpe_mult" ("B" = %B_shift : i32, "O" = %O_shift : i32) in_state(%inner_state_1) : !acc2.state<"snax_hwpe_mult">
//                           only loop-dependent variables left ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:       %tok = "acc2.launch"(%s_new_1) <{"param_names" = [], "accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
// CHECK-NEXT:       "acc2.await"(%tok) : (!acc2.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:       scf.yield %s_new_1 : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     }
// CHECK-NEXT:     %final = acc2.setup on "snax_hwpe_mult" ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32) in_state(%res_state) : !acc2.state<"snax_hwpe_mult">
//                                                size parameter can be inferred as unchanged and deleted ^
// CHECK-NEXT:     %token2 = "acc2.launch"(%final) <{"param_names" = [], "accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token<"snax_hwpe_mult">
// CHECK-NEXT:     "acc2.await"(%token2) : (!acc2.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }


func.func @nested_loops(%A : i32, %lb : i32, %ub : i32, %step : i32) {
  %0 = acc2.setup on "simple" () : !acc2.state<"simple">
  %1 = scf.for %i = %lb to %ub step %step iter_args(%2 = %0) -> (!acc2.state<"simple">) : i32 {
    %3 = scf.for %j = %i to %ub step %step iter_args(%4 = %2) -> (!acc2.state<"simple">) : i32 {
      %out_state = acc2.setup on "simple" ("A" = %A : i32, "i" = %i : i32, "j" = %j : i32) in_state(%4) : !acc2.state<"simple">
      scf.yield %out_state : !acc2.state<"simple">
    }
    scf.yield %3 : !acc2.state<"simple">
  }
  func.return
}

// check that every loop-nest only contains values that cannot be set at a higher level
// CHECK-NEXT:  func.func @nested_loops(%A_1 : i32, %lb_2 : i32, %ub_2 : i32, %step_2 : i32) {
// CHECK-NEXT:    %21 = acc2.setup on "simple" ("A" = %A_1 : i32) : !acc2.state<"simple">
//                                              ^^^^^^^^^^^^^^^^
// CHECK-NEXT:    %22 = scf.for %i = %lb_2 to %ub_2 step %step_2 iter_args(%23 = %21) -> (!acc2.state<"simple">) : i32 {
// CHECK-NEXT:      %24 = acc2.setup on "simple" ("i" = %i : i32) in_state(%23) : !acc2.state<"simple">
//                                                ^^^^^^^^^^^^^^
// CHECK-NEXT:      %25 = scf.for %j = %i to %ub_2 step %step_2 iter_args(%26 = %24) -> (!acc2.state<"simple">) : i32 {
// CHECK-NEXT:        %out_state = acc2.setup on "simple" ("j" = %j : i32) in_state(%26) : !acc2.state<"simple">
//                                                         ^^^^^^^^^^^^^^^
// CHECK-NEXT:        scf.yield %out_state : !acc2.state<"simple">
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %25 : !acc2.state<"simple">
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }


func.func @nested_loops_edge_cases(%A : i32, %lb : i32, %ub : i32, %step : i32) {
  %0 = acc2.setup on "simple" () : !acc2.state<"simple">
  %1 = scf.for %i = %lb to %ub step %step iter_args(%2 = %0) -> (!acc2.state<"simple">) : i32 {
    %3 = scf.for %j = %i to %ub step %step iter_args(%l0 = %2) -> (!acc2.state<"simple">) : i32 {
      %l1 = acc2.setup on "simple" ("A" = %A : i32, "B" = %A : i32, "i" = %i : i32, "j" = %j : i32) in_state(%l0) : !acc2.state<"simple">
      // test.op so that the two setups aren't fused
      "test.op"() : () -> ()
      %l2 = acc2.setup on "simple" ("A" = %A : i32, "B" = %ub : i32, "i" = %i : i32, "j" = %j : i32) in_state(%l1) : !acc2.state<"simple">
      // test op here too so that we don't accidentally dp some inter-iteration fusion
      "test.op"() : () -> ()
      scf.yield %l2 : !acc2.state<"simple">
    }
    scf.yield %3 : !acc2.state<"simple">
  }
  func.return
}

// check that B cannot be hoisted outside of the loop because it is set to two different loop invariant values
// CHECK-NEXT:  func.func @nested_loops_edge_cases(%A_2 : i32, %lb_3 : i32, %ub_3 : i32, %step_3 : i32) {
// CHECK-NEXT:    %27 = acc2.setup on "simple" ("A" = %A_2 : i32) : !acc2.state<"simple">
// CHECK-NEXT:    %28 = scf.for %i_1 = %lb_3 to %ub_3 step %step_3 iter_args(%29 = %27) -> (!acc2.state<"simple">) : i32 {
// CHECK-NEXT:      %30 = acc2.setup on "simple" ("i" = %i_1 : i32) in_state(%29) : !acc2.state<"simple">
// CHECK-NEXT:      %31 = scf.for %j_1 = %i_1 to %ub_3 step %step_3 iter_args(%l0 = %30) -> (!acc2.state<"simple">) : i32 {
// CHECK-NEXT:        %l1 = acc2.setup on "simple" ("B" = %A_2 : i32, "j" = %j_1 : i32) in_state(%l0) : !acc2.state<"simple">
// CHECK-NEXT:        "test.op"() : () -> ()
// CHECK-NEXT:        %l2 = acc2.setup on "simple" ("B" = %ub_3 : i32) in_state(%l1) : !acc2.state<"simple">
//                                                  ^^^^^^^^^^^ reset B, "j" can stay the same though
// CHECK-NEXT:        "test.op"() : () -> ()
// CHECK-NEXT:        scf.yield %l2 : !acc2.state<"simple">
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %31 : !acc2.state<"simple">
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }


func.func @setup_fusion(%A: i32, %B: i32) {
    %0 = acc2.setup on "simple" ("A" = %A : i32, "B" = %B : i32) : !acc2.state<"simple">
    %1 = acc2.setup on "simple" ("A" = %B : i32) in_state(%0) : !acc2.state<"simple">
    return
}

// check that the two ops are fused and the values of the later one overwrite the earlier ones
// CHECK-NEXT:  func.func @setup_fusion(%A_3 : i32, %B_1 : i32) {
// CHECK-NEXT:    %32 = acc2.setup on "simple" ("A" = %B_1 : i32, "B" = %B_1 : i32) : !acc2.state<"simple">
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }
