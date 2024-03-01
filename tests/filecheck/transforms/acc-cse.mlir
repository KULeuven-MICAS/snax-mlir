// RUN: snax-opt -p acc-cse

func.func @test() {
    %one, %two = "test.op"() : () -> (i32, i32)

    %state = "acc.setup"(%one, %two) <{
        param_names = ["A", "B"],
        accelerator = "acc1",
        operandSegmentSizes = array<i32: 2, 0>
    }> : (i32, i32) -> !acc.state<"acc1">

    %token = "acc.launch"() <{accelerator = "acc1"}>: () -> !acc.token

    %state2 = "acc.setup"(%one, %one, %state) <{
        param_names = ["A", "B"],
        accelerator = "acc1",
        operandSegmentSizes = array<i32: 2, 1>
    }> : (i32, i32, !acc.state<"acc1">) -> !acc.state<"acc1">

    %state3 = "acc.setup"(%one, %one, %state2) <{
        param_names = ["A", "B"],
        accelerator = "acc1",
        operandSegmentSizes = array<i32: 2, 1>
    }> : (i32, i32, !acc.state<"acc1">) -> !acc.state<"acc1">

    "acc.await"(%token) : (!acc.token) -> ()

    "test.op"(%state3) : (!acc.state<"acc1">) -> ()

    func.return
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func @test() {
// CHECK-NEXT:     %one, %two = "test.op"() : () -> (i32, i32)
// CHECK-NEXT:     %state = "acc.setup"(%one, %two) <{"param_names" = ["A", "B"], "accelerator" = "acc1", "operandSegmentSizes" = array<i32: 2, 0>}> : (i32, i32) -> !acc.state<"acc1">
// CHECK-NEXT:     %token = "acc.launch"() <{"accelerator" = "acc1"}> : () -> !acc.token
// CHECK-NEXT:     "acc.await"(%token) : (!acc.token) -> ()
// CHECK-NEXT:     "test.op"(%state) : (!acc.state<"acc1">) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
