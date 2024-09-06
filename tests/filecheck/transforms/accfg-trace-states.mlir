//RUN: ./compiler/snax-opt -p accfg-trace-states %s | filecheck %s

// CHECK-NEXT: builtin.module {



func.func @simple() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)

    %s1 = accfg.setup "snax_hwpe_mult" to (
        "A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32
    ) : !accfg.state<"snax_hwpe_mult">

    %s2 = accfg.setup "snax_hwpe_mult" to (
        "A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32
    ) : !accfg.state<"snax_hwpe_mult">

    return
}

// check that %s1 is added as an input to the second setup op
// CHECK-NEXT:  func.func @simple() {
// CHECK-NEXT:    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %s1 = accfg.setup "snax_hwpe_mult" to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:    %s2 = accfg.setup "snax_hwpe_mult" from %s1 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
//                                                                                                                                    ^^^^^^^^^^^^^
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }



func.func @double_acc_simple() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)

    %s1 = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">

    %s2 = "accfg.setup"(%B, %A, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult_2", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult_2">

    %s1_n = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">

    return
}

// check that the two different accelerator states have not been mixed up:
// CHECK-NEXT:  func.func @double_acc_simple() {
// CHECK-NEXT:    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %s1 = accfg.setup "snax_hwpe_mult" to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:    %s2 = accfg.setup "snax_hwpe_mult_2" to ("A" = %B : i32, "B" = %A : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult_2">
//                                                      ^ no added state argument, as this is the first "snax_hwpe_mult_2" setup
// CHECK-NEXT:    %s1_n = accfg.setup "snax_hwpe_mult" from %s1 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
//                                                     ^^^^^^^^^^ state of the first setup call here
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }



func.func @simple_if() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
    %cond = "test.op"() : ()  -> (i1)

    %s1 = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">

    "scf.if"(%cond) ({
        %s2 = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">
        yield
    }, {
        %s2 = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">
        yield
    }) : (i1) -> ()

    return
}

// check that %s1 is properly added to both ops, and that the result is woven into the if:
// CHECK-NEXT:  func.func @simple_if() {
// CHECK-NEXT:    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %cond = "test.op"() : () -> i1
// CHECK-NEXT:    %s1 = accfg.setup "snax_hwpe_mult" to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:    %0 = scf.if %cond -> (!accfg.state<"snax_hwpe_mult">) {
//                ^^                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:      %s2 = accfg.setup "snax_hwpe_mult" from %s1 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:      scf.yield %s2 : !accfg.state<"snax_hwpe_mult">
//                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %s2_1 = accfg.setup "snax_hwpe_mult" from %s1 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:      scf.yield %s2_1 : !accfg.state<"snax_hwpe_mult">
//                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:    }{}
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }


func.func @simple_if_double_acc() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
    %cond = "test.op"() : ()  -> (i1)

    %s1 = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">

    %s2 = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult_2", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult_2">

    "scf.if"(%cond) ({
        %s1_new = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">
        yield
    }, {
        %s2_new = "accfg.setup"(%B, %A, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult_2", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult_2">

        %s1_new = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">
        yield
    }) : (i1) -> ()

    return
}

// check that we didn't confuse one state with the other:
// CHECK-NEXT: func.func @simple_if_double_acc() {
// CHECK-NEXT:     %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:     %cond = "test.op"() : () -> i1
// CHECK-NEXT:     %s1 = accfg.setup "snax_hwpe_mult" to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:     %s2 = accfg.setup "snax_hwpe_mult_2" to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult_2">
//                                                                                                                                                    ^ no added state
// CHECK-NEXT:     %0, %1 = scf.if %cond -> (!accfg.state<"snax_hwpe_mult">, !accfg.state<"snax_hwpe_mult_2">) {
// CHECK-NEXT:       %s1_new = accfg.setup "snax_hwpe_mult" from %s1 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:       scf.yield %s1_new, %s2 : !accfg.state<"snax_hwpe_mult">, !accfg.state<"snax_hwpe_mult_2">
//                                      ^^^ yield outer state                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ correct type
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %s2_new = accfg.setup "snax_hwpe_mult_2" from %s2 to ("A" = %B : i32, "B" = %A : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult_2">
// CHECK-NEXT:       %s1_new_1 = accfg.setup "snax_hwpe_mult" from %s1 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:       scf.yield %s1_new_1, %s2_new : !accfg.state<"snax_hwpe_mult">, !accfg.state<"snax_hwpe_mult_2">
//                                        ^^^^^^^ yield inner state                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ correct type
// CHECK-NEXT:     }{}
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }

func.func @nested_if() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
    %cond, %cond2 = "test.op"() : ()  -> (i1, i1)

    %s1 = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">

    %thing = "scf.if"(%cond) ({
        %s2 = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">

        %c = "scf.if"(%cond2) ({
            %s3 = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">
            yield %A : i32
        }, {
            %s4 = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">
            yield %B : i32
        }) : (i1) -> (i32)

        %s3 = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">

        yield %c : i32
    }, {
        %s2 = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">
        yield %A : i32
    }) : (i1) -> (i32)

    return
}

// check that:
//  - The correct state is set in both nest levels of the if-else blocks
//  - The correct state is yielded from the blocks
//  - The existing yields are extended not replaced
// CHECK-NEXT: func.func @nested_if() {
// CHECK-NEXT:     %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:     %cond, %cond2 = "test.op"() : () -> (i1, i1)
// CHECK-NEXT:     %s1 = accfg.setup "snax_hwpe_mult" to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:     %thing, %thing_1 = scf.if %cond -> (i32, !accfg.state<"snax_hwpe_mult">) {
//                         ^^^^^^^^                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:       %s2 = accfg.setup "snax_hwpe_mult" from %s1 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
//                                                        ^^^^^^^^^^
// CHECK-NEXT:       %c, %c_1 = scf.if %cond2 -> (i32, !accfg.state<"snax_hwpe_mult">) {
//                       ^^^^                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:         %s3 = accfg.setup "snax_hwpe_mult" from %s2 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
//                                                        ^^^^^^^^^^
// CHECK-NEXT:         scf.yield %A, %s3 : i32, !accfg.state<"snax_hwpe_mult">
//                                   ^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %s4 = accfg.setup "snax_hwpe_mult" from %s2 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
//                                                        ^^^^^^^^^^
// CHECK-NEXT:         scf.yield %B, %s4 : i32, !accfg.state<"snax_hwpe_mult">
//                                  ^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:       }{}
// CHECK-NEXT:       %s3_1 = accfg.setup "snax_hwpe_mult" from %c_1 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
//                                                        ^^^^^^^^^
// CHECK-NEXT:       scf.yield %c, %s3_1 : i32, !accfg.state<"snax_hwpe_mult">
//                                 ^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %s2_1 = accfg.setup "snax_hwpe_mult" from %s1 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
//                                                        ^^^^^^^^
// CHECK-NEXT:       scf.yield %A, %s2_1 : i32, !accfg.state<"snax_hwpe_mult">
//                                 ^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:     }{}
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }

func.func @simple_loop() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)

    %lb, %ub, %step, %carry = "test.op"() : () -> (i32, i32, i32, i32)
    %res = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %carry) -> (i32) : i32{

        %s2 = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">

        yield %arg0 : i32
    }

    return
}

// check that an empty setup was added before the loop,
// and a new loop-carried state variable is added to the scf.for:
// CHECK-NEXT:  func.func @simple_loop() {
// CHECK-NEXT:    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %lb, %ub, %step, %carry = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %0 = accfg.setup "snax_hwpe_mult" to () : !accfg.state<"snax_hwpe_mult">
//                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ empty state
// CHECK-NEXT:    %res, %1 = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %carry, %2 = %0) -> (i32, !accfg.state<"snax_hwpe_mult">) : i32 {
//                      ^^                                                                ^^^^^^^^           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:      %s2 = accfg.setup "snax_hwpe_mult" from %2 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
//                                                     ^^^^^^^
// CHECK-NEXT:      scf.yield %arg0, %s2 : i32, !accfg.state<"snax_hwpe_mult">
//                                   ^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }



func.func @nested_loop() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)

    %s1 = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">

    %lb, %ub, %step, %carry = "test.op"() : () -> (i32, i32, i32, i32)
    %res = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %carry) -> (i32) : i32 {
        scf.for %y = %lb to %ub step %step : i32 {
            %s2 = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">
            yield
        }
        yield %arg0 : i32
    }

    return
}

// check that both loops receive a new loop-carried variable:
// CHECK-NEXT:  func.func @nested_loop() {
// CHECK-NEXT:    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %s1 = accfg.setup "snax_hwpe_mult" to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:    %lb, %ub, %step, %carry = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %res, %0 = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %carry, %1 = %s1) -> (i32, !accfg.state<"snax_hwpe_mult">) : i32 {
//                        ^^                                                                            ^^^^^^^^^^           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:      %2 = scf.for %y = %lb to %ub step %step iter_args(%3 = %1) -> (!accfg.state<"snax_hwpe_mult">) : i32 {
//                  ^^                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:        %s2 = accfg.setup "snax_hwpe_mult" from %3 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
//                                                       ^^^^^^^
// CHECK-NEXT:        scf.yield %s2 : !accfg.state<"snax_hwpe_mult">
//                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %arg0, %2 : i32, !accfg.state<"snax_hwpe_mult">
//                                   ^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }



func.func @loop_with_multiple_input_states() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)

    %s1 = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"snax_hwpe_mult">

    %sb = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "b", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"b">

    %lb, %ub, %step, %carry = "test.op"() : () -> (i32, i32, i32, i32)

    %res = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %carry) -> (i32) : i32 {
        %s2b = "accfg.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "b", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !accfg.state<"b">
        yield %arg0 : i32
    }

    return
}

// check that only accelerator "b" is passed into the loop:
// CHECK-NEXT:  func.func @loop_with_multiple_input_states() {
// CHECK-NEXT:    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %s1 = accfg.setup "snax_hwpe_mult" to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:    %sb = accfg.setup "b" to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"b">
// CHECK-NEXT:    %lb, %ub, %step, %carry = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %res, %0 = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %carry, %1 = %sb) -> (i32, !accfg.state<"b">) : i32 {
//                                                                                        ^^^^^^^^^           ^^^^^^^^^^^^^^^^
// CHECK-NEXT:      %s2b = accfg.setup "b" from %1 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"b">
// CHECK-NEXT:      scf.yield %arg0, %s2b : i32, !accfg.state<"b">
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }


func.func @random_parent_op() {
    "test.op"() ({
        %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)

        %s1 = accfg.setup "snax_hwpe_mult" to (
            "A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32
        ) : !accfg.state<"snax_hwpe_mult">

        %s2 = accfg.setup "snax_hwpe_mult" to (
            "A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32
        ) : !accfg.state<"snax_hwpe_mult">

        "test.termop"() : () -> ()
    }) : () -> ()

    return
}

// check that states have been connected inside the test op:
// CHECK-NEXT:  func.func @random_parent_op() {
// CHECK-NEXT:    "test.op"() ({
// CHECK-NEXT:      %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:      %s1 = accfg.setup "snax_hwpe_mult" to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:      %s2 = accfg.setup "snax_hwpe_mult" from %s1 to ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !accfg.state<"snax_hwpe_mult">
//                                                     ^^^^^^^^ perfect!
// CHECK-NEXT:      "test.termop"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }
