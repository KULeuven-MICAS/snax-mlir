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


// -----

func.func @scf_if_1(%i : i32, %cond: i1) {
    %state = accfg.setup "acc" to ("i" = %i : i32) : !accfg.state<"acc">

    "scf.if"(%cond) ({
        %0 = "accfg.launch"(%state) <{"param_names" = [], "accelerator" = "acc"}> : (!accfg.state<"acc">) -> !accfg.token<"acc">
        "accfg.await"(%0) : (!accfg.token<"acc">) -> ()

        yield
    }, {
        %1 = "accfg.launch"(%state) <{"param_names" = [], "accelerator" = "acc"}> : (!accfg.state<"acc">) -> !accfg.token<"acc">
        "accfg.await"(%1) : (!accfg.token<"acc">) -> ()

        yield
    }) : (i1) -> ()

    return
}

// CHECK-LABEL: @scf_if_1
// CHECK-NEXT:  [[state:%\S+]] = accfg.setup "acc" to ("i" = %i : i32) : !accfg.state<"acc">
// CHECK:       "accfg.launch"([[state]])
// CHECK-NEXT:  accfg.reset [[state]] : !accfg.state<"acc">
// CHECK:       yield
// CHECK:       "accfg.launch"([[state]])
// CHECK-NEXT:  accfg.reset [[state]] : !accfg.state<"acc">
// CHECK:       yield


// -----

func.func @scf_if_2(%i : i32, %cond: i1) {
    %state = accfg.setup "acc" to ("i" = %i : i32) : !accfg.state<"acc">

    %2 = "scf.if"(%cond) ({
        %0 = "accfg.launch"(%state) <{"param_names" = [], "accelerator" = "acc"}> : (!accfg.state<"acc">) -> !accfg.token<"acc">
        "accfg.await"(%0) : (!accfg.token<"acc">) -> ()

        yield %state : !accfg.state<"acc">
    }, {
        %1 = "accfg.launch"(%state) <{"param_names" = [], "accelerator" = "acc"}> : (!accfg.state<"acc">) -> !accfg.token<"acc">
        "accfg.await"(%1) : (!accfg.token<"acc">) -> ()

        yield %state : !accfg.state<"acc">
    }) : (i1) -> (!accfg.state<"acc">)

    return
}

// CHECK-LABEL: @scf_if_2
// CHECK-NEXT:  [[state:%\S+]] = accfg.setup "acc" to ("i" = %i : i32) : !accfg.state<"acc">
// CHECK:       [[state2:%\S+]] = "scf.if"
// CHECK:       "accfg.launch"([[state]])
// CHECK-NOT:   accfg.reset [[state]] : !accfg.state<"acc">
// CHECK:       yield
// CHECK:       "accfg.launch"([[state]])
// CHECK-NOT:   accfg.reset [[state]] : !accfg.state<"acc">
// CHECK:       yield
// CHECK:       accfg.reset [[state2]] : !accfg.state<"acc">


// -----

func.func @simple_sequence(%A: i32) {
    %s1 = accfg.setup "acc" to ("A" = %A : i32) : !accfg.state<"acc">

    %s2 = accfg.setup "acc" from %s1 to ("A" = %A : i32) : !accfg.state<"acc">

    return
}

// CHECK-LABEL: @simple_sequence
// CHECK-NEXT: %s1 = accfg.setup "acc" to ("A" = %A : i32) : !accfg.state<"acc">
// CHECK-NEXT: %s2 = accfg.setup "acc" from %s1 to ("A" = %A : i32) : !accfg.state<"acc">
// CHECK-NEXT: accfg.reset %s2
