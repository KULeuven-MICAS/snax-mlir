// RUN: snax-opt --split-input-file -p accfg-insert-resets %s | filecheck %s --check-prefixes=CHECK,BEFORE
// RUN: snax-opt --split-input-file -p "accfg-insert-resets{reset-after-await=true}" %s | filecheck %s --check-prefixes=CHECK,AFTER

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
// CHECK-NOT:   accfg.reset [[state]]


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
// CHECK-NEXT:  [[state:%\S+]] = accfg.setup "acc" to ("i" = %i : i32) : !accfg.state<"acc">
// CHECK-NEXT:  [[token:%\S+]] = "accfg.launch"(%state)
// BEFORE-NEXT: accfg.reset [[state]] : !accfg.state<"acc">
// CHECK-NEXT:  "accfg.await"([[token]])
// AFTER-NEXT:  accfg.reset [[state]] : !accfg.state<"acc">


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
// BEFORE-NEXT: accfg.reset [[state]] : !accfg.state<"acc">
// CHECK-NEXT:  "accfg.await"
// AFTER-NEXT:  accfg.reset [[state]] : !accfg.state<"acc">
// CHECK:       yield
// CHECK:       "accfg.launch"([[state]])
// BEFORE-NEXT: accfg.reset [[state]] : !accfg.state<"acc">
// CHECK-NEXT:  "accfg.await"
// AFTER-NEXT:  accfg.reset [[state]] : !accfg.state<"acc">
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


// -----

func.func @simple_loop() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
    %lb, %ub, %step, %carry = "test.op"() : () -> (i32, i32, i32, i32)
    %0 = accfg.setup "snax_hwpe_mult" to () : !accfg.state<"snax_hwpe_mult">
    %res, %1 = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %carry, %2 = %0) -> (i32, !accfg.state<"snax_hwpe_mult">) : i32 {
        %s2 = accfg.setup "snax_hwpe_mult" from %2 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
        %t = "accfg.launch"(%s2) <{accelerator = "snax_hwpe_mult", param_names = []}> : (!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
        "accfg.await"(%t) : (!accfg.token<"snax_hwpe_mult">) -> ()
        scf.yield %arg0, %s2 : i32, !accfg.state<"snax_hwpe_mult">
    }
    func.return
}

// CHECK-LABEL: func.func @simple_loop() {
// CHECK-NEXT:    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %lb, %ub, %step, %carry = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %0 = accfg.setup "snax_hwpe_mult" to () : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:    %res, %1 = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %carry, %2 = %0) -> (i32, !accfg.state<"snax_hwpe_mult">) : i32 {
// CHECK-NEXT:      %s2 = accfg.setup "snax_hwpe_mult" from %2 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:      %t = "accfg.launch"(%s2) <{"accelerator" = "snax_hwpe_mult", "param_names" = []}> : (!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:      "accfg.await"(%t) : (!accfg.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:      scf.yield %arg0, %s2 : i32, !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:    }
// CHECK-NEXT:    accfg.reset %1 : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }
