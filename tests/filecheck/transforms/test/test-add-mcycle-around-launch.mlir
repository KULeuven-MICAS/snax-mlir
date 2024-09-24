// RUN: snax-opt --split-input-file -p test-add-mcycle-around-launch %s | filecheck %s

"accfg.accelerator"() <{
    name               = @acc1,
    fields             = {A=0x3c0, B=0x3c1},
    launch_fields      = {launch=0x3cf},
    barrier            = 0x7c3
}> : () -> ()

func.func @test() {
    %one, %two = "test.op"() : () -> (i32, i32)
    %zero = arith.constant 0 : i32
    %state = "accfg.setup"(%one, %two) <{
        param_names = ["A", "B"],
        accelerator = "acc1",
        operandSegmentSizes = array<i32: 2, 0>
    }> {"test_attr" = 100 : i64} : (i32, i32) -> !accfg.state<"acc1">

    %token = "accfg.launch"(%zero, %state) <{
        param_names = ["launch"],
        accelerator = "acc1"
    }>: (i32, !accfg.state<"acc1">) -> !accfg.token<"acc1">

    "accfg.await"(%token) : (!accfg.token<"acc1">) -> ()

    func.return
}


// CHECK: snax.mcycle
// CHECK-NEXT: accfg.launch
// CHECK-NEXT: accfg.await
// CHECK-NEXT: snax.mcycle
