// RUN: snax-opt --split-input-file -p accfg-config-overlap %s | filecheck %s
// RUN: snax-opt --split-input-file -p accfg-config-overlap,accfg-insert-resets %s | filecheck %s --check-prefix CHECK --check-prefix RESET

func.func @simple(%A: i32, %B: i32) {
    %s1 = accfg.setup "simple" to ("A" = %A : i32) : !accfg.state<"simple">
    %t = "accfg.launch"(%s1) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
    "accfg.await"(%t) : (!accfg.token<"simple">) -> ()

    %s2 = accfg.setup "simple" from %s1 to ("A" = %B : i32) : !accfg.state<"simple">

    return
}

// check that simple overlapping works
// CHECK:       func.func @simple(%A : i32, %B : i32) {
// CHECK-NEXT:    %s1 = accfg.setup "simple" to ("A" = %A : i32) : !accfg.state<"simple">
// CHECK-NEXT:    %t = "accfg.launch"(%s1) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
//                      ∨∨∨∨∨ setup comes right after launch ∧∧∧∧∧
// CHECK-NEXT:    %s2 = accfg.setup "simple" from %s1 to ("A" = %B : i32) : !accfg.state<"simple">
// reset right after setup
// RESET-NEXT:    accfg.reset %s2 : !accfg.state<"simple">
// CHECK-NEXT:    "accfg.await"(%t) : (!accfg.token<"simple">) -> ()
//                 ∧∧∧∧∧∧∧∧∧∧∧ await now after next setup
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }



// -----

func.func @computed(%A: i32) {
    %s1 = accfg.setup "simple" to ("A" = %A : i32) : !accfg.state<"simple">
    %t = "accfg.launch"(%s1) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
    "accfg.await"(%t) : (!accfg.token<"simple">) -> ()

    %c = arith.constant 64 : i32
    %A_plus = arith.addi %A, %c : i32
    %irrelevant = arith.constant 42 : i32
    "test.op"(%irrelevant) : (i32) -> ()
    %s2 = accfg.setup "simple" from %s1 to ("A" = %A_plus : i32) : !accfg.state<"simple">

    return
}

// check that values needed to calculate the setup values are also moved
// CHECK:       func.func @computed(%A : i32) {
// CHECK-NEXT:    %s1 = accfg.setup "simple" to ("A" = %A : i32) : !accfg.state<"simple">
// CHECK-NEXT:    %t = "accfg.launch"(%s1) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
//                ∨∨∨∨∨ arith values moved up
// CHECK-NEXT:    %c = arith.constant 64 : i32
// CHECK-NEXT:    %A_plus = arith.addi %A, %c : i32
//                                                              ∨∨∨∨∨∨∨ this SSA value is valid (the ops that calculate it have been moved)
// CHECK-NEXT:    %s2 = accfg.setup "simple" from %s1 to ("A" = %A_plus : i32) : !accfg.state<"simple">
// RESET-NEXT:    accfg.reset %s2 : !accfg.state<"simple">
//                ∨∨∨∨∨ await pushed down
// CHECK-NEXT:    "accfg.await"(%t) : (!accfg.token<"simple">) -> ()
// CHECK-NEXT:    %irrelevant = arith.constant 42 : i32
// CHECK-NEXT:    "test.op"(%irrelevant) : (i32) -> ()
//                ∧∧∧∧∧∧∧∧∧∧∧ Irrelevant constant was not moved up
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }


// -----

func.func @simple_negative(%A: i32, %B: i32, %i1: i1) {
    %s1 = accfg.setup "simple" to ("A" = %A : i32) : !accfg.state<"simple">
    %t = "accfg.launch"(%s1) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
    "accfg.await"(%t) : (!accfg.token<"simple">) -> ()

    "scf.if"(%i1) ({
        %s2 = accfg.setup "simple" from %s1 to ("A" = %B : i32) : !accfg.state<"simple">
        scf.yield
    }, {}) : (i1) -> ()

    return
}

// check that we don't move setups out of control flow (as this would change observable behaviour)
// CHECK:       func.func @simple_negative(%A : i32, %B : i32, %i1 : i1) {
// CHECK-NEXT:    %s1 = accfg.setup "simple" to ("A" = %A : i32) : !accfg.state<"simple">
// CHECK-NEXT:    %t = "accfg.launch"(%s1) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
// CHECK-NEXT:    "accfg.await"(%t) : (!accfg.token<"simple">) -> ()
// CHECK-NEXT:    scf.if %i1 {
// CHECK-NEXT:      %s2 = accfg.setup "simple" from %s1 to ("A" = %B : i32) : !accfg.state<"simple">
//                        ∧∧∧∧∧∧∧∧∧∧∧ yep, op is still here!
// RESET-NEXT:      accfg.reset %s2 : !accfg.state<"simple">
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }


// -----

func.func @single_loop(%A : i32, %lb : i32, %ub : i32, %step : i32) {
  %0 = accfg.setup "simple" to () : !accfg.state<"simple">

  %1 = scf.for %i = %lb to %ub step %step iter_args(%l0 = %0) -> (!accfg.state<"simple">) : i32 {

    %l1 = accfg.setup "simple" from %l0 to ("A" = %A : i32, "B" = %A : i32, "i" = %i : i32) : !accfg.state<"simple">
    %t = "accfg.launch"(%l1) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
    "accfg.await"(%t) : (!accfg.token<"simple">) -> ()

    scf.yield %l1 : !accfg.state<"simple">
  }

  func.return
}

// CHECK:        func.func @single_loop(%A : i32, %lb : i32, %ub : i32, %step : i32) {
// CHECK-NEXT:     %0 = accfg.setup "simple" to () : !accfg.state<"simple">
// CHECK-NEXT:     %l1 = accfg.setup "simple" from %0 to ("A" = %A : i32, "B" = %A : i32, "i" = %lb : i32) : !accfg.state<"simple">
// CHECK-NEXT:     %1 = scf.for %i = %lb to %ub step %step iter_args(%l0 = %l1) -> (!accfg.state<"simple">) : i32 {
// CHECK-NEXT:       %t = "accfg.launch"(%l0) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
// CHECK-NEXT:       %2 = arith.addi %i, %step : i32
// CHECK-NEXT:       %l1_1 = accfg.setup "simple" from %l0 to ("A" = %A : i32, "B" = %A : i32, "i" = %2 : i32) : !accfg.state<"simple">
// CHECK-NEXT:       "accfg.await"(%t) : (!accfg.token<"simple">) -> ()
// CHECK-NEXT:       scf.yield %l1_1 : !accfg.state<"simple">
// CHECK-NEXT:     }
// RESET-NEXT:     accfg.reset %1 : !accfg.state<"simple">
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }


// -----

func.func @complex_loop(%A : i32, %lb : i32, %ub : i32, %step : i32) {
  %0 = accfg.setup "simple" to () : !accfg.state<"simple">
  %c2_i32 = arith.constant 2 : i32

  %1 = scf.for %i = %lb to %ub step %step iter_args(%l0 = %0) -> (!accfg.state<"simple">) : i32 {
    %b = arith.addi %i, %c2_i32 : i32
    %l1 = accfg.setup "simple" from %l0 to ("A" = %A : i32, "B" = %b : i32, "i" = %i : i32) : !accfg.state<"simple">
    %t = "accfg.launch"(%l1) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
    "accfg.await"(%t) : (!accfg.token<"simple">) -> ()

    scf.yield %l1 : !accfg.state<"simple">
  }

  func.return
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @complex_loop(%A : i32, %lb : i32, %ub : i32, %step : i32) {
// CHECK-NEXT:     %0 = accfg.setup "simple" to () : !accfg.state<"simple">
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %b = arith.addi %lb, %c2_i32 : i32
//                                 ∧∧∧∧∧∧∧∧∧∧∧∧ loop 0 calculation with loop variable replaced by %lb
// CHECK-NEXT:     %l1 = accfg.setup "simple" from %0 to ("A" = %A : i32, "B" = %b : i32, "i" = %lb : i32) : !accfg.state<"simple">
//                       ∧∧∧∧∧∧∧∧∧∧∧∧ loop 0 setup with correct vars            ∧∧              ∧∧
// CHECK-NEXT:     %1 = scf.for %i = %lb to %ub step %step iter_args(%l0 = %l1) -> (!accfg.state<"simple">) : i32 {
//                   %b_1 = arith.addi %i, %c2_i32 : i32
//                   ∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧ this is dead now and can be eliminated by dce
// CHECK-NEXT:       %t = "accfg.launch"(%l0) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
// CHECK-NEXT:       %2 = arith.addi %i, %step : i32
//                        ∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧ compute next loops %i
// CHECK-NEXT:       %b_1 = arith.addi %2, %c2_i32 : i32
//                          ∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧ additional calculations done on the loop variable (now takes b_1 after dce)
// CHECK-NEXT:       %l1_1 = accfg.setup "simple" from %l0 to ("A" = %A : i32, "B" = %b_1 : i32, "i" = %2 : i32) : !accfg.state<"simple">
//                           ∧∧∧∧∧ setup before await ∨∨∨∨∨
// CHECK-NEXT:       "accfg.await"(%t) : (!accfg.token<"simple">) -> ()
// CHECK-NEXT:       scf.yield %l1_1 : !accfg.state<"simple">
// CHECK-NEXT:     }
// RESET-NEXT:     accfg.reset %1 : !accfg.state<"simple">
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }


// -----

func.func @nested_loops(%A : i32, %lb : i32, %ub : i32, %step : i32) {
  %0 = accfg.setup "simple" to () : !accfg.state<"simple">
  %1 = scf.for %i = %lb to %ub step %step iter_args(%2 = %0) -> (!accfg.state<"simple">) : i32 {
    %3 = scf.for %j = %i to %ub step %step iter_args(%4 = %2) -> (!accfg.state<"simple">) : i32 {

      %out_state = accfg.setup "simple" from %4 to ("A" = %A : i32, "i" = %i : i32, "j" = %j : i32) : !accfg.state<"simple">
      %t = "accfg.launch"(%out_state) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
      "accfg.await"(%t) : (!accfg.token<"simple">) -> ()

      scf.yield %out_state : !accfg.state<"simple">
    }
    scf.yield %3 : !accfg.state<"simple">
  }
  func.return
}

// note that accfg-dedup has not been ran yet on this IR
// we check that we correctly overlap setup/await in the inner loop
// CHECK:       func.func @nested_loops(%A : i32, %lb : i32, %ub : i32, %step : i32) {
// CHECK-NEXT:    %0 = accfg.setup "simple" to () : !accfg.state<"simple">
// CHECK-NEXT:    %1 = scf.for %i = %lb to %ub step %step iter_args(%2 = %0) -> (!accfg.state<"simple">) : i32 {
// CHECK-NEXT:      %out_state = accfg.setup "simple" from %2 to ("A" = %A : i32, "i" = %i : i32, "j" = %i : i32) : !accfg.state<"simple">
//                               ∧∧∧∧∧∧∧∧∧∧∧ loop 0 setup with correct vars             ∧∧              ∧∧
// CHECK-NEXT:      %3 = scf.for %j = %i to %ub step %step iter_args(%out_state_1 = %out_state) -> (!accfg.state<"simple">) : i32 {
// CHECK-NEXT:        %t = "accfg.launch"(%out_state_1) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
// CHECK-NEXT:        %4 = arith.addi %j, %step : i32
//                         ∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧∧ calculate next loops var
// CHECK-NEXT:        %out_state_2 = accfg.setup "simple" from %out_state_1 to ("A" = %A : i32, "i" = %i : i32, "j" = %4 : i32) : !accfg.state<"simple">
//                                   ∧∧∧∧∧∧∧∧∧∧∧ next loops setup                                                   ∧∧
// CHECK-NEXT:        "accfg.await"(%t) : (!accfg.token<"simple">) -> ()
// CHECK-NEXT:        scf.yield %out_state_2 : !accfg.state<"simple">
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %3 : !accfg.state<"simple">
// CHECK-NEXT:    }
// RESET-NEXT:    accfg.reset %1 : !accfg.state<"simple">
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }


// -----

func.func @double_setup_loop(%A : i32, %B : i32, %lb : i32, %ub : i32, %step : i32) {
  %0 = accfg.setup "simple" to () : !accfg.state<"simple">
  %c2 = arith.constant 2 : i32

  %1 = scf.for %i = %lb to %ub step %step iter_args(%l0 = %0) -> (!accfg.state<"simple">) : i32 {

    // first setup, sets up %A
    %i_plus_2 = arith.addi %i, %c2 : i32
    %l1 = accfg.setup "simple" from %l0 to ("A" = %A : i32, "B" = %A : i32, "i" = %i_plus_2 : i32) : !accfg.state<"simple">
    %t1 = "accfg.launch"(%l1) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
    "accfg.await"(%t1) : (!accfg.token<"simple">) -> ()

    // second setup, sets up %B
    %i2 = arith.addi %i, %i : i32
    %l2 = accfg.setup "simple" from %l1 to ("A" = %B : i32, "B" = %B : i32, "i" = %i2 : i32) : !accfg.state<"simple">
    %t2 = "accfg.launch"(%l2) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
    "accfg.await"(%t2) : (!accfg.token<"simple">) -> ()

    scf.yield %l2 : !accfg.state<"simple">
  }

  func.return
}

// Check that the loop structure is preserved and only a single setup is moved up.
// We expect the loop to contain a launch, setup, await, launch, setup, await sequence.
// We also expect that the first setup that sets up %A is pulled out of the loop.
// CHECK:       func.func @double_setup_loop(%A : i32, %B : i32, %lb : i32, %ub : i32, %step : i32) {
// CHECK-NEXT:    %0 = accfg.setup "simple" to () : !accfg.state<"simple">
// CHECK-NEXT:    %c2 = arith.constant 2 : i32
//                  Correctly adding variables
// CHECK-NEXT:    %i_plus = arith.addi %lb, %c2 : i32
//                  Setup for %A (pulled out)
// CHECK-NEXT:    %l1 = accfg.setup "simple" from %0 to ("A" = %A : i32, "B" = %A : i32, "i" = %i_plus : i32) : !accfg.state<"simple">
// CHECK-NEXT:    %1 = scf.for %i = %lb to %ub step %step iter_args(%l0 = %l1) -> (!accfg.state<"simple">) : i32 {
//                  This operation is DCE'd
//                    vvvvvvvvvvvvvvvvvv
//                  %i_plus_1 = arith.addi %i, %c2 : i32
//                      launch
// CHECK-NEXT:      %t1 = "accfg.launch"(%l0) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
// CHECK-NEXT:      %i2 = arith.addi %i, %i : i32
//                      setup
// CHECK-NEXT:      %l2 = accfg.setup "simple" from %l0 to ("A" = %B : i32, "B" = %B : i32, "i" = %i2 : i32) : !accfg.state<"simple">
//                      await
// CHECK-NEXT:      "accfg.await"(%t1) : (!accfg.token<"simple">) -> ()
//                      launch
// CHECK-NEXT:      %t2 = "accfg.launch"(%l2) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
// CHECK-NEXT:      %2 = arith.addi %i, %step : i32
//                  because of DCE, this is renamed to i_plus_1, instead of i_plus_2
// CHECK-NEXT:      %i_plus_1 = arith.addi %2, %c2 : i32
//                      setup
// CHECK-NEXT:      %l1_1 = accfg.setup "simple" from %l2 to ("A" = %A : i32, "B" = %A : i32, "i" = %i_plus_1 : i32) : !accfg.state<"simple">
//                      await
// CHECK-NEXT:      "accfg.await"(%t2) : (!accfg.token<"simple">) -> ()
// CHECK-NEXT:      scf.yield %l1_1 : !accfg.state<"simple">
// CHECK-NEXT:    }
// RESET-NEXT:    accfg.reset %1 : !accfg.state<"simple">
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }
