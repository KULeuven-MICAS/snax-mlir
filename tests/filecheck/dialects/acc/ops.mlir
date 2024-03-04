// RUN: XDSL_ROUNDTRIP

acc.accelerator @acc1 {"A" = 42 : i32, "B" = 69 : i32, "C" = 420 : i32}

func.func @test() {
    %one, %two = "test.op"() : () -> (i32, i32)

    %state = "acc.setup"(%one, %two) <{
        param_names = ["A", "B"],
        accelerator = "acc1",
        operandSegmentSizes = array<i32: 2, 0>
    }> : (i32, i32) -> !acc.state<"acc1">

    %token = "acc.launch"(%state) <{accelerator = "acc1"}>: (!acc.state<"acc1">) -> !acc.token

    %state2 = "acc.setup"(%one, %two, %state) <{
        param_names = ["A", "B"],
        accelerator = "acc1",
        operandSegmentSizes = array<i32: 2, 1>
    }> : (i32, i32, !acc.state<"acc1">) -> !acc.state<"acc1">

    "acc.await"(%token) : (!acc.token) -> ()

    func.return
}


// CHECK-NEXT: "builtin.module"() ({
// CHECK-NEXT:   "acc.accelerator"() <{"name" = @acc1, "fields" = {"A" = 42 : i32, "B" = 69 : i32, "C" = 420 : i32}}>
// CHECK-NEXT:   "func.func"() <{"sym_name" = "test", "function_type" = () -> ()}> ({
// CHECK-NEXT:     %one, %two = "test.op"() : () -> (i32, i32)
// CHECK-NEXT:     %state = "acc.setup"(%one, %two) <{"param_names" = ["A", "B"], "accelerator" = "acc1", "operandSegmentSizes" = array<i32: 2, 0>}> : (i32, i32) -> !acc.state<"acc1">
// CHECK-NEXT:     %token = "acc.launch"(%state) <{"accelerator" = "acc1"}> : (!acc.state<"acc1">) -> !acc.token
// CHECK-NEXT:     %state2 = "acc.setup"(%one, %two, %state) <{"param_names" = ["A", "B"], "accelerator" = "acc1", "operandSegmentSizes" = array<i32: 2, 1>}> : (i32, i32, !acc.state<"acc1">) -> !acc.state<"acc1">
// CHECK-NEXT:     "acc.await"(%token) : (!acc.token) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()
