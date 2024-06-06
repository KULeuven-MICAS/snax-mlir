// RUN: snax-opt --split-input-file -p accfg-config-overlap %s | filecheck %s

func.func @simple(%A: i32, %B: i32) {
    %s1 = accfg.setup "simple" to ("A" = %A : i32) : !accfg.state<"simple">
    %t = "accfg.launch"(%s1) <{param_names = [], accelerator = "simple"}> : (!accfg.state<"simple">) -> !accfg.token<"simple">
    "accfg.await"(%t) : (!accfg.token<"simple">) -> ()

    %s2 = accfg.setup "simple" from %s1 to ("A" = %B : i32) : !accfg.state<"simple">

    return
}

// check that simple merging works
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

func.func @simple(%A: i32, %B: i32, %i1: i1) {
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
// CHECK:       func.func @simple(%A : i32, %B : i32, %i1 : i1) {
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
