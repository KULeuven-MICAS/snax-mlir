// RUN: snax-opt --split-input-file -p test-add-mcycle-around-launch %s | filecheck %s

"accfg.accelerator"() <{"name" = @acc1, "fields" = {"A" = 960 : i64, "B" = 961 : i64}, "launch_fields" = {"launch" = 975 : i64}, "barrier" = 1987 : i64}> : () -> ()
%one, %two = "test.op"() : () -> (i32, i32)
%zero = arith.constant 0 : i32
%state = accfg.setup "acc1" to ("A" = %one : i32, "B" = %two : i32) attrs {"test_attr" = 100 : i64} : !accfg.state<"acc1">
%token = "accfg.launch"(%zero, %state) <{"param_names" = ["launch"], "accelerator" = "acc1"}> : (i32, !accfg.state<"acc1">) -> !accfg.token<"acc1">
"accfg.await"(%token) : (!accfg.token<"acc1">) -> ()


// CHECK: snax.mcycle
// CHECK-NEXT: accfg.launch
// CHECK-NEXT: accfg.await
// CHECK-NEXT: snax.mcycle
