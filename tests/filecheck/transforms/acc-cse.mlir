// RUN: snax-opt -p acc-cse

func.func @test() {
    %one, %two = "test.op"() : () -> (i32, i32)

    %state = "acc.setup"(%one, %two) <{
        param_names = ["A", "B"],
        accelerator = "acc1",
        operandSegmentSizes = array<i32: 2, 0>
    }> : (i32, i32) -> !acc.state<"acc1">

    %token = "acc.launch"() <{accelerator = "acc1"}>: () -> !acc.token

    %state2 = "acc.setup"(%one, %two, %state) <{
        param_names = ["A", "B"],
        accelerator = "acc1",
        operandSegmentSizes = array<i32: 2, 1>
    }> : (i32, i32, !acc.state<"acc1">) -> !acc.state<"acc1">

    "acc.await"(%token) : (!acc.token) -> ()

    func.return
}
