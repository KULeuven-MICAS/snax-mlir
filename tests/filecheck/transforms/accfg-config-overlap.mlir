// RUN: snax-opt --split-input-file -p accfg-config-overlap %s | filecheck %s

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
// CHECK-NEXT:    %t = "accfg.launch"(%s1) <{"param_names" = [], "accelerator" = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
//                      ∨∨∨∨∨ setup comes right after launch ∧∧∧∧∧
// CHECK-NEXT:    %s2 = accfg.setup "simple" from %s1 to ("A" = %B : i32) : !accfg.state<"simple">
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
    %s2 = accfg.setup "simple" from %s1 to ("A" = %A_plus : i32) : !accfg.state<"simple">

    return
}

// check that values needed to calculate the setup values are also moved
// CHECK:       func.func @computed(%A : i32) {
// CHECK-NEXT:    %s1 = accfg.setup "simple" to ("A" = %A : i32) : !accfg.state<"simple">
// CHECK-NEXT:    %t = "accfg.launch"(%s1) <{"param_names" = [], "accelerator" = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
//                ∨∨∨∨∨ arith values moved up
// CHECK-NEXT:    %c = arith.constant 64 : i32
// CHECK-NEXT:    %A_plus = arith.addi %A, %c : i32
//                                                              ∨∨∨∨∨∨∨ this SSA value is valid (the ops that calculate it have been moved)
// CHECK-NEXT:    %s2 = accfg.setup "simple" from %s1 to ("A" = %A_plus : i32) : !accfg.state<"simple">
//                ∨∨∨∨∨ await pushed down
// CHECK-NEXT:    "accfg.await"(%t) : (!accfg.token<"simple">) -> ()
// CHECK-NEXT:    %irrelevant = arith.constant 42 : i32
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
// CHECK-NEXT:    %t = "accfg.launch"(%s1) <{"param_names" = [], "accelerator" = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
// CHECK-NEXT:    "accfg.await"(%t) : (!accfg.token<"simple">) -> ()
// CHECK-NEXT:    "scf.if"(%i1) ({
// CHECK-NEXT:      %s2 = accfg.setup "simple" from %s1 to ("A" = %B : i32) : !accfg.state<"simple">
//                        ∧∧∧∧∧∧∧∧∧∧∧ yep, op is still here!
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }, {
// CHECK-NEXT:    }) : (i1) -> ()
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
// CHECK-NEXT:     %1 = accfg.setup "simple" from %0 to ("A" = %A : i32, "B" = %A : i32, "i" = %lb : i32) : !accfg.state<"simple">
// CHECK-NEXT:     %2 = scf.for %i = %lb to %ub step %step iter_args(%l0 = %1) -> (!accfg.state<"simple">) : i32 {
// CHECK-NEXT:       %t = "accfg.launch"(%l0) <{"param_names" = [], "accelerator" = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
// CHECK-NEXT:       %3 = arith.addi %i, %step : i32
// CHECK-NEXT:       %4 = accfg.setup "simple" from %l0 to ("A" = %A : i32, "B" = %A : i32, "i" = %3 : i32) : !accfg.state<"simple">
// CHECK-NEXT:       "accfg.await"(%t) : (!accfg.token<"simple">) -> ()
// CHECK-NEXT:       scf.yield %4 : !accfg.state<"simple">
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
