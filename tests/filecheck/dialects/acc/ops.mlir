// RUN: XDSL_ROUNDTRIP

acc2.accelerator @acc1 {A=0x3c0, B=0x3c1}

func.func @test() {
    %one, %two = "test.op"() : () -> (i32, i32)

    %state = "acc2.setup"(%one, %two) <{
        param_names = ["A", "B"],
        accelerator = "acc1",
        operandSegmentSizes = array<i32: 2, 0>
    }> : (i32, i32) -> !acc2.state<"acc1">

    %token = "acc2.launch"(%state) <{accelerator = "acc1"}>: (!acc2.state<"acc1">) -> !acc2.token

    %state2 = "acc2.setup"(%one, %two, %state) <{
        param_names = ["A", "B"],
        accelerator = "acc1",
        operandSegmentSizes = array<i32: 2, 1>
    }> : (i32, i32, !acc2.state<"acc1">) -> !acc2.state<"acc1">

    "acc2.await"(%token) : (!acc2.token) -> ()

    func.return
}


// CHECK-NEXT: "builtin.module"() ({
// CHECK-NEXT:   "acc2.accelerator"() <{"name" = @acc1, "fields" = {"A" = 960 : i64, "B" = 961 : i64}}> : () -> ()
// CHECK-NEXT:   "func.func"() <{"sym_name" = "test", "function_type" = () -> ()}> ({
// CHECK-NEXT:     %one, %two = "test.op"() : () -> (i32, i32)
// CHECK-NEXT:     %state = "acc2.setup"(%one, %two) <{"param_names" = ["A", "B"], "accelerator" = "acc1", "operandSegmentSizes" = array<i32: 2, 0>}> : (i32, i32) -> !acc2.state<"acc1">
// CHECK-NEXT:     %token = "acc2.launch"(%state) <{"accelerator" = "acc1"}> : (!acc2.state<"acc1">) -> !acc2.token
// CHECK-NEXT:     %state2 = "acc2.setup"(%one, %two, %state) <{"param_names" = ["A", "B"], "accelerator" = "acc1", "operandSegmentSizes" = array<i32: 2, 1>}> : (i32, i32, !acc2.state<"acc1">) -> !acc2.state<"acc1">
// CHECK-NEXT:     "acc2.await"(%token) : (!acc2.token) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()
