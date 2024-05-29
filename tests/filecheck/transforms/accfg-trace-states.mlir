//RUN: ./compiler/snax-opt -p accfg-trace-states %s | filecheck %s

// CHECK-NEXT: builtin.module {



func.func @simple() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)

    %s1 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">

    %s2 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">

    return
}

// check that %s1 is added as an input to the second setup op
// CHECK-NEXT:  func.func @simple() {
// CHECK-NEXT:    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %s1 = acc2.setup on "snax_hwpe_mult" ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:    %s2 = acc2.setup on "snax_hwpe_mult" ("A" = %A : i32, "B" = %B : i32, "O" = %O : i32, "nr_iters" = %nr_iters : i32) in_state(%s1) : !acc2.state<"snax_hwpe_mult">
//                                                                                                                                    ^^^^^^^^^^^^^
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }



func.func @double_acc_simple() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)

    %s1_1 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">

    %s2_1 = "acc2.setup"(%B, %A, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult_2", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult_2">

    %s1_2 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">

    return
}

// check that the two different accelerator states have not been mixed up:
// CHECK-NEXT:  func.func @double_acc_simple() {
// CHECK-NEXT:    %A_1, %B_1, %O_1, %nr_iters_1 = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %s1_1 = acc2.setup on "snax_hwpe_mult" ("A" = %A_1 : i32, "B" = %B_1 : i32, "O" = %O_1 : i32, "nr_iters" = %nr_iters_1 : i32) : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:    %s2_1 = acc2.setup on "snax_hwpe_mult_2" ("A" = %B_1 : i32, "B" = %A_1 : i32, "O" = %O_1 : i32, "nr_iters" = %nr_iters_1 : i32) : !acc2.state<"snax_hwpe_mult_2">
//                                                                        no added state argument, as this is the first "snax_hwpe_mult_2" setup ^
// CHECK-NEXT:    %s1_2 = acc2.setup on "snax_hwpe_mult" ("A" = %A_1 : i32, "B" = %B_1 : i32, "O" = %O_1 : i32, "nr_iters" = %nr_iters_1 : i32) in_state(%s1_1) : !acc2.state<"snax_hwpe_mult">
//                                                                                                           state of the first setup call here ^^^^^^^^^^^^^^^
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }



func.func @simple_if() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
    %cond = "test.op"() : ()  -> (i1)

    %s1 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">

    "scf.if"(%cond) ({
        %s2 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">
        yield
    }, {
        %s2 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">
        yield
    }) : (i1) -> ()

    return
}

// check that %s1 is properly added to both ops, and that the result is woven into the if:
// CHECK-NEXT:  func.func @simple_if() {
// CHECK-NEXT:    %A_2, %B_2, %O_2, %nr_iters_2 = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %cond = "test.op"() : () -> i1
// CHECK-NEXT:    %s1_1 = acc2.setup on "snax_hwpe_mult" ("A" = %A_2 : i32, "B" = %B_2 : i32, "O" = %O_2 : i32, "nr_iters" = %nr_iters_2 : i32) : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:    %0 = "scf.if"(%cond) ({
//                ^^
// CHECK-NEXT:      %s2_1 = acc2.setup on "snax_hwpe_mult" ("A" = %A_2 : i32, "B" = %B_2 : i32, "O" = %O_2 : i32, "nr_iters" = %nr_iters_2 : i32) in_state(%s1_1) : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:      scf.yield %s2_1 : !acc2.state<"snax_hwpe_mult">
//                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %s2_2 = acc2.setup on "snax_hwpe_mult" ("A" = %A_2 : i32, "B" = %B_2 : i32, "O" = %O_2 : i32, "nr_iters" = %nr_iters_2 : i32) in_state(%s1_1) : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:      scf.yield %s2_2 : !acc2.state<"snax_hwpe_mult">
//                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:    }) : (i1) -> !acc2.state<"snax_hwpe_mult">
//                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }



func.func @simple_if_double_acc() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
    %cond = "test.op"() : ()  -> (i1)

    %s1_1 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">

    %s2_1 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult_2", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult_2">

    "scf.if"(%cond) ({
        %s1_2 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">
        yield
    }, {
        %s2_2 = "acc2.setup"(%B, %A, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult_2", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult_2">

        %s1_2 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">
        yield
    }) : (i1) -> ()

    return
}

// check that we didn't confuse one state with the other:
// CHECK-NEXT:  func.func @simple_if_double_acc() {
// CHECK-NEXT:    %A_3, %B_3, %O_3, %nr_iters_3 = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %cond_1 = "test.op"() : () -> i1
// CHECK-NEXT:       %s1_1_1 = acc2.setup on "snax_hwpe_mult" ("A" = %A_3 : i32, "B" = %B_3 : i32, "O" = %O_3 : i32, "nr_iters" = %nr_iters_3 : i32) : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:       %s2_1_1 = acc2.setup on "snax_hwpe_mult_2" ("A" = %A_3 : i32, "B" = %B_3 : i32, "O" = %O_3 : i32, "nr_iters" = %nr_iters_3 : i32) : !acc2.state<"snax_hwpe_mult_2">
//                                                                                                                                                    ^ no added state
// CHECK-NEXT:    %1, %2 = "scf.if"(%cond_1) ({
// CHECK-NEXT:      %s1_2_1 = acc2.setup on "snax_hwpe_mult" ("A" = %A_3 : i32, "B" = %B_3 : i32, "O" = %O_3 : i32, "nr_iters" = %nr_iters_3 : i32) in_state(%s1_1_1) : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:      scf.yield %s1_2_1, %s2_1_1 : !acc2.state<"snax_hwpe_mult">, !acc2.state<"snax_hwpe_mult_2">
//                                     ^^^^^^^ yield outer state                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ correct type
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %s2_2 = acc2.setup on "snax_hwpe_mult_2" ("A" = %B_3 : i32, "B" = %A_3 : i32, "O" = %O_3 : i32, "nr_iters" = %nr_iters_3 : i32) in_state(%s2_1_1) : !acc2.state<"snax_hwpe_mult_2">
// CHECK-NEXT:      %s1_2_2 = acc2.setup on "snax_hwpe_mult" ("A" = %A_3 : i32, "B" = %B_3 : i32, "O" = %O_3 : i32, "nr_iters" = %nr_iters_3 : i32) in_state(%s1_1_1) : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:      scf.yield %s1_2_2, %s2_2 : !acc2.state<"snax_hwpe_mult">, !acc2.state<"snax_hwpe_mult_2">
//                                     ^^^^^ yield inner state                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ correct type
// CHECK-NEXT:    }) : (i1) -> (!acc2.state<"snax_hwpe_mult">, !acc2.state<"snax_hwpe_mult_2">)
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }



func.func @nested_if() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)
    %cond, %cond2 = "test.op"() : ()  -> (i1, i1)

    %s1 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">

    %thing = "scf.if"(%cond) ({
        %s2 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">

        %c = "scf.if"(%cond2) ({
            %s3 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">
            yield %A : i32
        }, {
            %s4 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">
            yield %B : i32
        }) : (i1) -> (i32)

        %s3 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">

        yield %c : i32
    }, {
        %s2 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">
        yield %A : i32
    }) : (i1) -> (i32)

    return
}

// check that:
//  - The correct state is set in both nest levels of the if-else blocks
//  - The correct state is yielded from the blocks
//  - The existing yields are extended not replaced
// CHECK-NEXT:  func.func @nested_if() {
// CHECK-NEXT:    %A_4, %B_4, %O_4, %nr_iters_4 = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %cond_2, %cond2 = "test.op"() : () -> (i1, i1)
// CHECK-NEXT:    %s1_2 = acc2.setup on "snax_hwpe_mult" ("A" = %A_4 : i32, "B" = %B_4 : i32, "O" = %O_4 : i32, "nr_iters" = %nr_iters_4 : i32) : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:    %thing, %thing_1 = "scf.if"(%cond_2) ({
//                        ^^^^^^^^
// CHECK-NEXT:      %s2_3 = acc2.setup on "snax_hwpe_mult" ("A" = %A_4 : i32, "B" = %B_4 : i32, "O" = %O_4 : i32, "nr_iters" = %nr_iters_4 : i32) in_state(%s1_2) : !acc2.state<"snax_hwpe_mult">
//                                                                                                                                                ^^^^^^^^^^^^^^^
// CHECK-NEXT:      %c, %c_1 = "scf.if"(%cond2) ({
// CHECK-NEXT:        %s3 = acc2.setup on "snax_hwpe_mult" ("A" = %A_4 : i32, "B" = %B_4 : i32, "O" = %O_4 : i32, "nr_iters" = %nr_iters_4 : i32) in_state(%s2_3) : !acc2.state<"snax_hwpe_mult">
//                                                                                                                                                ^^^^^^^^^^^^^^^
// CHECK-NEXT:        scf.yield %A_4, %s3 : i32, !acc2.state<"snax_hwpe_mult">
//                                    ^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %s4 = acc2.setup on "snax_hwpe_mult" ("A" = %A_4 : i32, "B" = %B_4 : i32, "O" = %O_4 : i32, "nr_iters" = %nr_iters_4 : i32) in_state(%s2_3) : !acc2.state<"snax_hwpe_mult">
//                                                                                                                                                ^^^^^^^^^^^^^^^
// CHECK-NEXT:        scf.yield %B_4, %s4 : i32, !acc2.state<"snax_hwpe_mult">
//                                    ^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:      }) : (i1) -> (i32, !acc2.state<"snax_hwpe_mult">)
// CHECK-NEXT:      %s3_1 = acc2.setup on "snax_hwpe_mult" ("A" = %A_4 : i32, "B" = %B_4 : i32, "O" = %O_4 : i32, "nr_iters" = %nr_iters_4 : i32) in_state(%c_1) : !acc2.state<"snax_hwpe_mult">
//                                                                                                                                                ^^^^^^^^^^^^^^^
// CHECK-NEXT:      scf.yield %c, %s3_1 : i32, !acc2.state<"snax_hwpe_mult">
//                                ^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %s2_4 = acc2.setup on "snax_hwpe_mult" ("A" = %A_4 : i32, "B" = %B_4 : i32, "O" = %O_4 : i32, "nr_iters" = %nr_iters_4 : i32) in_state(%s1_2) : !acc2.state<"snax_hwpe_mult">
//                                                                                                                                                ^^^^^^^^^^^^^^^
// CHECK-NEXT:      scf.yield %A_4, %s2_4 : i32, !acc2.state<"snax_hwpe_mult">
//                                  ^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:    }) : (i1) -> (i32, !acc2.state<"snax_hwpe_mult">)
//                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }



func.func @simple_loop() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)

    %lb, %ub, %step, %carry = "test.op"() : () -> (i32, i32, i32, i32)
    %res = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %carry) -> (i32) {

        %s2 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">

        yield %arg0 : i32
    }

    return
}

// check that an empty setup was added before the loop,
// and a new loop-carried state variable is added to the scf.for:
// CHECK-NEXT:  func.func @simple_loop() {
// CHECK-NEXT:    %A_5, %B_5, %O_5, %nr_iters_5 = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %lb, %ub, %step, %carry = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %3 = acc2.setup on "snax_hwpe_mult" () : !acc2.state<"snax_hwpe_mult">
//                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ empty state
// CHECK-NEXT:    %res, %4 = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %carry, %5 = %3) -> (i32, !acc2.state<"snax_hwpe_mult">) : i32 {
//                      ^^                                                                ^^^^^^^^           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:      %s2_5 = acc2.setup on "snax_hwpe_mult" ("A" = %A_5 : i32, "B" = %B_5 : i32, "O" = %O_5 : i32, "nr_iters" = %nr_iters_5 : i32) in_state(%5) : !acc2.state<"snax_hwpe_mult">
//                                                                                                                                                ^^^^^^^^^^^^
// CHECK-NEXT:      scf.yield %arg0, %s2_5 : i32, !acc2.state<"snax_hwpe_mult">
//                                   ^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }



func.func @nested_loop() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)

    %s1 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">

    %lb, %ub, %step, %carry = "test.op"() : () -> (i32, i32, i32, i32)
    %res = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %carry) -> (i32) {
        scf.for %y = %lb to %ub step %step {
            %s2 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">
            yield
        }
        yield %arg0 : i32
    }

    return
}

// check that both loops receive a new loop-carried variable:
// CHECK-NEXT:  func.func @nested_loop() {
// CHECK-NEXT:    %A_6, %B_6, %O_6, %nr_iters_6 = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %s1_3 = acc2.setup on "snax_hwpe_mult" ("A" = %A_6 : i32, "B" = %B_6 : i32, "O" = %O_6 : i32, "nr_iters" = %nr_iters_6 : i32) : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:    %lb_1, %ub_1, %step_1, %carry_1 = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %res_1, %6 = scf.for %i_1 = %lb_1 to %ub_1 step %step_1 iter_args(%arg0_1 = %carry_1, %7 = %s1_3) -> (i32, !acc2.state<"snax_hwpe_mult">) : i32 {
//                        ^^                                                                            ^^^^^^^^^^           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:      %8 = scf.for %y = %lb_1 to %ub_1 step %step_1 iter_args(%9 = %7) -> (!acc2.state<"snax_hwpe_mult">) : i32 {
//                  ^^                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:        %s2_6 = acc2.setup on "snax_hwpe_mult" ("A" = %A_6 : i32, "B" = %B_6 : i32, "O" = %O_6 : i32, "nr_iters" = %nr_iters_6 : i32) in_state(%9) : !acc2.state<"snax_hwpe_mult">
//                                                                                                                                                  ^^^^^^^^^^^^
// CHECK-NEXT:        scf.yield %s2_6 : !acc2.state<"snax_hwpe_mult">
//                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %arg0_1, %8 : i32, !acc2.state<"snax_hwpe_mult">
//                                     ^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }



func.func @loop_with_multiple_input_states() {
    %A, %B, %O, %nr_iters = "test.op"() : () -> (i32, i32, i32, i32)

    %s1 = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"snax_hwpe_mult">

    %sb = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "b", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"b">

    %lb, %ub, %step, %carry = "test.op"() : () -> (i32, i32, i32, i32)

    %res = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %carry) -> (i32) {
        %s2b = "acc2.setup"(%A, %B, %O, %nr_iters) <{"accelerator" = "b", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "nr_iters"]}> : (i32, i32, i32, i32) -> !acc2.state<"b">
        yield %arg0 : i32
    }

    return
}

// check that only accelerator "b" is passed into the loop:
// CHECK-NEXT:  func.func @loop_with_multiple_input_states() {
// CHECK-NEXT:    %A_7, %B_7, %O_7, %nr_iters_7 = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %s1_4 = acc2.setup on "snax_hwpe_mult" ("A" = %A_7 : i32, "B" = %B_7 : i32, "O" = %O_7 : i32, "nr_iters" = %nr_iters_7 : i32) : !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:    %sb = acc2.setup on "b" ("A" = %A_7 : i32, "B" = %B_7 : i32, "O" = %O_7 : i32, "nr_iters" = %nr_iters_7 : i32) : !acc2.state<"b">
// CHECK-NEXT:    %lb_2, %ub_2, %step_2, %carry_2 = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %res_2, %10 = scf.for %i_2 = %lb_2 to %ub_2 step %step_2 iter_args(%arg0_2 = %carry_2, %11 = %sb) -> (i32, !acc2.state<"b">) : i32 {
//                                                                                                       ^^^^^^^^^           ^^^^^^^^^^^^^^^^
// CHECK-NEXT:      %s2b = acc2.setup on "b" ("A" = %A_7 : i32, "B" = %B_7 : i32, "O" = %O_7 : i32, "nr_iters" = %nr_iters_7 : i32) in_state(%11) : !acc2.state<"b">
// CHECK-NEXT:      scf.yield %arg0_2, %s2b : i32, !acc2.state<"b">
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }
