// RUN: ./compiler/snax-opt %s -p accfg-dedup | filecheck %s

func.func public @simple_mult(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xi32>) {
  %0 = arith.constant 0 : index
  %cst = arith.constant 0 : i5
  %1 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
  %2 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
  %3 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi32>) -> index
  %4 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index

  %5 = "accfg.setup"(%1, %2, %3, %4) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index) -> !accfg.state<"snax_hwpe_mult">

  %6 = "accfg.launch"(%cst, %5) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5,!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
  "accfg.await"(%6) : (!accfg.token<"snax_hwpe_mult">) -> ()

  %7 = "test.op"() : () -> i1

  %8, %9 = "scf.if"(%7) ({
    %10 = "accfg.setup"(%1, %3, %3, %4, %5) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 1>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index, !accfg.state<"snax_hwpe_mult">) -> !accfg.state<"snax_hwpe_mult">

    %11 = "accfg.launch"(%cst, %10) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5,!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
    "accfg.await"(%11) : (!accfg.token<"snax_hwpe_mult">) -> ()

    %12 = "test.op"() : () -> i32
    scf.yield %12, %10 : i32, !accfg.state<"snax_hwpe_mult">
  }, {
    %13 = "test.op"() : () -> i32

    %14 = "accfg.setup"(%1, %2, %3, %4, %5) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 1>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index, !accfg.state<"snax_hwpe_mult">) -> !accfg.state<"snax_hwpe_mult">
    %15 = "accfg.launch"(%cst, %14) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5,!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
    "accfg.await"(%15) : (!accfg.token<"snax_hwpe_mult">) -> ()

    scf.yield %13, %14 : i32, !accfg.state<"snax_hwpe_mult">
  }) : (i1) -> (i32, !accfg.state<"snax_hwpe_mult">)

  "test.op"(%8) : (i32) -> ()

  %16 = "accfg.setup"(%1, %3, %2, %4, %9) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 1>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index, !accfg.state<"snax_hwpe_mult">) -> !accfg.state<"snax_hwpe_mult">
  %17 = "accfg.launch"(%cst, %16) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5,!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
  "accfg.await"(%17) : (!accfg.token<"snax_hwpe_mult">) -> ()


  %lb = arith.constant 0 : index
  %ub = arith.constant 100 : index
  %step = arith.constant 1 : index

  %res = scf.for %iv = %lb to %ub step %step iter_args(%inner_state = %16) -> (!accfg.state<"snax_hwpe_mult">) {

    %s_new = "accfg.setup"(%1, %2, %3, %iv, %inner_state) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 1>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index, !accfg.state<"snax_hwpe_mult">) -> !accfg.state<"snax_hwpe_mult">
    %222 = "accfg.launch"(%cst, %s_new) <{"param_names" = ["launch"], "accelerator" = "snax_hwpe_mult"}> : (i5,!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
    "accfg.await"(%222) : (!accfg.token<"snax_hwpe_mult">) -> ()
    scf.yield %s_new : !accfg.state<"snax_hwpe_mult">
  }
  func.return
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xi32>) {
// CHECK-NEXT:     %0 = arith.constant 0 : index
// CHECK-NEXT:     %cst = arith.constant 0 : i5
// CHECK-NEXT:     %1 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
// CHECK-NEXT:     %2 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
// CHECK-NEXT:     %3 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi32>) -> index
// CHECK-NEXT:     %4 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index
// CHECK-NEXT:     %5 = accfg.setup "snax_hwpe_mult" to ("A" = %1 : index, "B" = %2 : index, "O" = %3 : index, "size" = %4 : index) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:     %6 = "accfg.launch"(%cst, %5) <{param_names = ["launch"], accelerator = "snax_hwpe_mult"}> : (i5, !accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:     "accfg.await"(%6) : (!accfg.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:     %7 = "test.op"() : () -> i1
// CHECK-NEXT:   %8, %9 = scf.if %7 -> (i32, !accfg.state<"snax_hwpe_mult">) {
// CHECK-NEXT:     %10 = accfg.setup "snax_hwpe_mult" from %5 to ("B" = %3 : index) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:     %11 = "accfg.launch"(%cst, %10) <{param_names = ["launch"], accelerator = "snax_hwpe_mult"}> : (i5, !accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:     "accfg.await"(%11) : (!accfg.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:     %12 = "test.op"() : () -> i32
// CHECK-NEXT:     %13 = accfg.setup "snax_hwpe_mult" from %10 to ("O" = %2 : index) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:     scf.yield %12, %13 : i32, !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %14 = "test.op"() : () -> i32
// CHECK-NEXT:     %15 = "accfg.launch"(%cst, %5) <{param_names = ["launch"], accelerator = "snax_hwpe_mult"}> : (i5, !accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:     "accfg.await"(%15) : (!accfg.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:     %16 = accfg.setup "snax_hwpe_mult" from %5 to ("B" = %3 : index, "O" = %2 : index) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:     scf.yield %14, %16 : i32, !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:   }
// CHECK-NEXT:     "test.op"(%8) : (i32) -> ()
// CHECK-NEXT:     %17 = "accfg.launch"(%cst, %9) <{param_names = ["launch"], accelerator = "snax_hwpe_mult"}> : (i5, !accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:     "accfg.await"(%17) : (!accfg.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:     %lb = arith.constant 0 : index
// CHECK-NEXT:     %ub = arith.constant 100 : index
// CHECK-NEXT:     %step = arith.constant 1 : index
// CHECK-NEXT:     %18 = accfg.setup "snax_hwpe_mult" from %9 to ("B" = %2 : index, "O" = %3 : index) : !accfg.state<"snax_hwpe_mult">
//                                 new setup op for invariant ops ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:     %res = scf.for %iv = %lb to %ub step %step iter_args(%inner_state = %18) -> (!accfg.state<"snax_hwpe_mult">) {
// CHECK-NEXT:        %s_new = accfg.setup "snax_hwpe_mult" from %inner_state to ("size" = %iv : index) : !accfg.state<"snax_hwpe_mult">
//                                                      only loop-dependent vars remaining ^^^
// CHECK-NEXT:       %19 = "accfg.launch"(%cst, %s_new) <{param_names = ["launch"], accelerator = "snax_hwpe_mult"}> : (i5, !accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:       "accfg.await"(%19) : (!accfg.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:       scf.yield %s_new : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }


func.func @scf_for_test(%A: i32, %B: i32) {
  %O = "test.op"() : () -> i32
  %c32 = arith.constant 32 : i32

  // initial launch
  %init = accfg.setup "snax_hwpe_mult" to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "size" = %c32 : i32) : !accfg.state<"snax_hwpe_mult">
  %token = "accfg.launch"(%init) <{"param_names" = [], "accelerator" = "snax_hwpe_mult"}> : (!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
  "accfg.await"(%token) : (!accfg.token<"snax_hwpe_mult">) -> ()

  // set up the loop
  %lb = arith.constant 0 : index
  %ub = arith.constant 100 : index
  %step = arith.constant 1 : index

  // set up one loop-invariant operand
  %A_shift = arith.addi %A, %c32 : i32

  // iterate a bunch of times
  %res_state = scf.for %iv = %lb to %ub step %step iter_args(%inner_state = %init) -> (!accfg.state<"snax_hwpe_mult">) {
    // some variables computed in-loop:
    %B_shift = arith.addi %B, %c32 : i32
    %O_shift = arith.addi %O, %c32 : i32

    // launch with loop-invariant and loop-dependent vars:
    %s_new = accfg.setup "snax_hwpe_mult" from %inner_state to ("A" = %A_shift : i32, "B" = %B_shift : i32, "O" = %O_shift : i32, "size" = %c32 : i32) : !accfg.state<"snax_hwpe_mult">
    %tok = "accfg.launch"(%s_new) <{"param_names" = [], "accelerator" = "snax_hwpe_mult"}> : (!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
    "accfg.await"(%tok) : (!accfg.token<"snax_hwpe_mult">) -> ()

    scf.yield %s_new : !accfg.state<"snax_hwpe_mult">
  }

  // tailing launch, with same inputs as initial launch:
  %final = accfg.setup "snax_hwpe_mult" from %res_state to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "size" = %c32 : i32) : !accfg.state<"snax_hwpe_mult">
  %token2 = "accfg.launch"(%final) <{"param_names" = [], "accelerator" = "snax_hwpe_mult"}> :  (!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
  "accfg.await"(%token2) : (!accfg.token<"snax_hwpe_mult">) -> ()
  return
}

// CHECK-NEXT:   func.func @scf_for_test(%A : i32, %B : i32) {
// CHECK-NEXT:     %O = "test.op"() : () -> i32
// CHECK-NEXT:     %c32 = arith.constant 32 : i32
// CHECK-NEXT:     %init = accfg.setup "snax_hwpe_mult" to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "size" = %c32 : i32) : !accfg.state<"snax_hwpe_mult">
//                 ^^^ first setup should be untouched ^^^
// CHECK-NEXT:     %token = "accfg.launch"(%init) <{param_names = [], accelerator = "snax_hwpe_mult"}> : (!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:     "accfg.await"(%token) : (!accfg.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:     %lb = arith.constant 0 : index
// CHECK-NEXT:     %ub = arith.constant 100 : index
// CHECK-NEXT:     %step = arith.constant 1 : index
// CHECK-NEXT:     %A_shift = arith.addi %A, %c32 : i32
// CHECK-NEXT:     %0 = accfg.setup "snax_hwpe_mult" from %init to ("A" = %A_shift : i32) : !accfg.state<"snax_hwpe_mult">
//                        new setup op for loop-invariant variables ^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:     %res_state = scf.for %iv = %lb to %ub step %step iter_args(%inner_state = %0) -> (!accfg.state<"snax_hwpe_mult">) {
// CHECK-NEXT:       %B_shift = arith.addi %B, %c32 : i32
// CHECK-NEXT:       %O_shift = arith.addi %O, %c32 : i32
// CHECK-NEXT:       %s_new = accfg.setup "snax_hwpe_mult" from %inner_state to ("B" = %B_shift : i32, "O" = %O_shift : i32) : !accfg.state<"snax_hwpe_mult">
//                                                only loop-dependent variables left ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:       %tok = "accfg.launch"(%s_new) <{param_names = [], accelerator = "snax_hwpe_mult"}> : (!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:       "accfg.await"(%tok) : (!accfg.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:       scf.yield %s_new : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:     }
// CHECK-NEXT:     %final = accfg.setup "snax_hwpe_mult" from %res_state to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32) : !accfg.state<"snax_hwpe_mult">
//                                                size parameter can be inferred as unchanged and deleted ^
// CHECK-NEXT:     %token2 = "accfg.launch"(%final) <{param_names = [], accelerator = "snax_hwpe_mult"}> : (!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:     "accfg.await"(%token2) : (!accfg.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }


func.func @nested_loops(%A : i32, %lb : i32, %ub : i32, %step : i32) {
  %0 = accfg.setup "simple" to () : !accfg.state<"simple">
  %1 = scf.for %i = %lb to %ub step %step iter_args(%2 = %0) -> (!accfg.state<"simple">) : i32 {
    %3 = scf.for %j = %i to %ub step %step iter_args(%4 = %2) -> (!accfg.state<"simple">) : i32 {
      %out_state = accfg.setup "simple" from %4 to ("A" = %A : i32, "i" = %i : i32, "j" = %j : i32) : !accfg.state<"simple">
      scf.yield %out_state : !accfg.state<"simple">
    }
    scf.yield %3 : !accfg.state<"simple">
  }
  func.return
}

// check that every loop-nest only contains values that cannot be set at a higher level
// CHECK-NEXT:  func.func @nested_loops(%A : i32, %lb : i32, %ub : i32, %step : i32) {
// CHECK-NEXT:    %0 = accfg.setup "simple" to ("A" = %A : i32) : !accfg.state<"simple">
//                                               ^^^^^^^^^^^^^^
// CHECK-NEXT:    %1 = scf.for %i = %lb to %ub step %step iter_args(%2 = %0) -> (!accfg.state<"simple">) : i32 {
// CHECK-NEXT:      %3 = accfg.setup "simple" from %2 to ("i" = %i : i32) : !accfg.state<"simple">
//                                                        ^^^^^^^^^^^^^^
// CHECK-NEXT:      %4 = scf.for %j = %i to %ub step %step iter_args(%5 = %3) -> (!accfg.state<"simple">) : i32 {
// CHECK-NEXT:        %out_state = accfg.setup "simple" from %5 to ("j" = %j : i32) : !accfg.state<"simple">
//                                                                  ^^^^^^^^^^^^^^
// CHECK-NEXT:        scf.yield %out_state : !accfg.state<"simple">
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %4 : !accfg.state<"simple">
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }


func.func @nested_loops_edge_cases(%A : i32, %lb : i32, %ub : i32, %step : i32) {
  %0 = accfg.setup "simple" to () : !accfg.state<"simple">
  %1 = scf.for %i = %lb to %ub step %step iter_args(%2 = %0) -> (!accfg.state<"simple">) : i32 {
    %3 = scf.for %j = %i to %ub step %step iter_args(%l0 = %2) -> (!accfg.state<"simple">) : i32 {
      %l1 = accfg.setup "simple" from %l0 to ("A" = %A : i32, "B" = %A : i32, "i" = %i : i32, "j" = %j : i32) : !accfg.state<"simple">
      // test.op so that the two setups aren't fused
      "test.op"() : () -> ()
      %l2 = accfg.setup "simple" from %l1 to ("A" = %A : i32, "B" = %ub : i32, "i" = %i : i32, "j" = %j : i32) : !accfg.state<"simple">
      // test op here too so that we don't accidentally dp some inter-iteration fusion
      "test.op"() : () -> ()
      scf.yield %l2 : !accfg.state<"simple">
    }
    scf.yield %3 : !accfg.state<"simple">
  }
  func.return
}

// check that B cannot be hoisted outside of the loop because it is set to two different loop invariant values
// CHECK-NEXT:  func.func @nested_loops_edge_cases(%A : i32, %lb : i32, %ub : i32, %step : i32) {
// CHECK-NEXT:    %0 = accfg.setup "simple" to ("A" = %A : i32) : !accfg.state<"simple">
// CHECK-NEXT:    %1 = scf.for %i = %lb to %ub step %step iter_args(%2 = %0) -> (!accfg.state<"simple">) : i32 {
// CHECK-NEXT:      %3 = accfg.setup "simple" from %2 to ("i" = %i : i32) : !accfg.state<"simple">
// CHECK-NEXT:      %4 = scf.for %j = %i to %ub step %step iter_args(%l0 = %3) -> (!accfg.state<"simple">) : i32 {
// CHECK-NEXT:        %l1 = accfg.setup "simple" from %l0 to ("B" = %A : i32, "j" = %j : i32) : !accfg.state<"simple">
// CHECK-NEXT:        "test.op"() : () -> ()
// CHECK-NEXT:        %l2 = accfg.setup "simple" from %l1 to ("B" = %ub : i32) : !accfg.state<"simple">
//                      reset B, "j" can stay the same though ^^^^^^^^^^^
// CHECK-NEXT:        "test.op"() : () -> ()
// CHECK-NEXT:        scf.yield %l2 : !accfg.state<"simple">
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %4 : !accfg.state<"simple">
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }


func.func @setup_fusion(%A: i32, %B: i32) {
    %0 = accfg.setup "simple" to ("A" = %A : i32, "B" = %B : i32) : !accfg.state<"simple">
    %1 = accfg.setup "simple" from %0 to ("A" = %B : i32) : !accfg.state<"simple">
    return
}

// check that the two ops are fused and the values of the later one overwrite the earlier ones
// CHECK-NEXT:  func.func @setup_fusion(%A : i32, %B : i32) {
// CHECK-NEXT:    %0 = accfg.setup "simple" to ("A" = %B : i32, "B" = %B : i32) : !accfg.state<"simple">
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }
