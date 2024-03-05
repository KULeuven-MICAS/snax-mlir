// RUN: ./compiler/snax-opt %s -p acc-dedup | filecheck %s

func.func @test() {
    %one, %two = "test.op"() : () -> (i32, i32)

    %state = "acc2.setup"(%one, %two) <{
        param_names = ["A", "B"],
        accelerator = "acc1",
        operandSegmentSizes = array<i32: 2, 0>
    }> : (i32, i32) -> !acc2.state<"acc1">

    %token = "acc2.launch"(%state) <{accelerator = "acc1"}> : (!acc2.state<"acc1">) -> !acc2.token

    %state2 = "acc2.setup"(%one, %one, %state) <{
        param_names = ["A", "B"],
        accelerator = "acc1",
        operandSegmentSizes = array<i32: 2, 1>
    }> : (i32, i32, !acc2.state<"acc1">) -> !acc2.state<"acc1">

    %state3 = "acc2.setup"(%one, %one, %state2) <{
        param_names = ["A", "B"],
        accelerator = "acc1",
        operandSegmentSizes = array<i32: 2, 1>
    }> : (i32, i32, !acc2.state<"acc1">) -> !acc2.state<"acc1">

    "acc2.await"(%token) : (!acc2.token) -> ()

    "test.op"(%state3) : (!acc2.state<"acc1">) -> ()

    func.return
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func @test() {
// CHECK-NEXT:     %one, %two = "test.op"() : () -> (i32, i32)
// CHECK-NEXT:     %state = "acc2.setup"(%one, %two) <{"param_names" = ["A", "B"], "accelerator" = "acc1", "operandSegmentSizes" = array<i32: 2, 0>}> : (i32, i32) -> !acc2.state<"acc1">
// CHECK-NEXT:     %token = "acc2.launch"(%state) <{"accelerator" = "acc1"}> : (!acc2.state<"acc1">) -> !acc2.token
// CHECK-NEXT:     %state2 = "acc2.setup"(%one, %state) <{"param_names" = ["B"], "accelerator" = "acc1", "operandSegmentSizes" = array<i32: 1, 1>}> : (i32, !acc2.state<"acc1">) -> !acc2.state<"acc1">
// CHECK-NEXT:     "acc2.await"(%token) : (!acc2.token) -> ()
// CHECK-NEXT:     "test.op"(%state2) : (!acc2.state<"acc1">) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
