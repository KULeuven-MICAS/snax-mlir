// RUN: snax-opt --split-input-file -p accfg-insert-resets %s | filecheck %s

func.func @args(%state: !accfg.state<"simple">) {
    return
}

// CHECK-LABEL: @args
// CHECK-SAME:  ([[state:%\S+]] : !accfg.state<"simple">)
// CHECK:       accfg.reset [[state]]


// -----

func.func @returned_args(%state: !accfg.state<"simple">) -> !accfg.state<"simple"> {
    return %state : !accfg.state<"simple">
}

// CHECK-LABEL: @returned_args
// CHECK-SAME:  ([[state:%\S+]] : !accfg.state<"simple">)
// CHECK-NOT:       accfg.reset [[state]]


// -----

func.func @simple(%i : i32) {
    %state = accfg.setup "acc" to ("i" = %i : i32) : !accfg.state<"acc">
    return
}

// CHECK-LABEL: @simple
// CHECK-NEXT: [[state:%\S+]] = accfg.setup "acc" to ("i" = %i : i32) : !accfg.state<"acc">
// CHECK:      accfg.reset [[state]] : !accfg.state<"acc">


// -----

func.func @with_uses(%i : i32) {
    %state = accfg.setup "acc" to ("i" = %i : i32) : !accfg.state<"acc">

    %t = "accfg.launch"(%state) <{"param_names" = [], "accelerator" = "acc"}> : (!accfg.state<"acc">) -> !accfg.token<"acc">
    "accfg.await"(%t) : (!accfg.token<"acc">) -> ()
    return
}

// CHECK-LABEL: @with_uses
// CHECK-NEXT: [[state:%\S+]] = accfg.setup "acc" to ("i" = %i : i32) : !accfg.state<"acc">
// CHECK-NEXT: [[token:%\S+]] = "accfg.launch"(%state)
// CHECK-NEXT: accfg.reset [[state]] : !accfg.state<"acc">
// CHECK-NEXT: "accfg.await"([[token]])
