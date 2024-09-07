// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK: builtin.module {
// CHECK-GENERIC: "builtin.module"() ({

%0 = "test.op"() : () -> i32
// CHECK-NEXT:   %0 = "test.op"() : () -> i32
// CHECK-NEXT-GENERIC:   %0 = "test.op"() : () -> i32

%1 = "kernel.add" (%0, %0) : (i32, i32) -> i32
// CHECK-NEXT:   %1 = kernel.add %0, %0 : i32, i32 -> i32
// CHECK-NEXT-GENERIC:   %1 = "kernel.add"(%0, %0) : (i32, i32) -> i32

%2 = "kernel.mul" (%0, %0) : (i32, i32) -> i32
// CHECK-NEXT:   %2 = kernel.mul %0, %0 : i32, i32 -> i32
// CHECK-NEXT-GENERIC:   %2 = "kernel.mul"(%0, %0) : (i32, i32) -> i32

%3 = "kernel.mac" (%0, %0, %0) : (i32, i32, i32) -> i32
// CHECK-NEXT:   %3 = kernel.mac %0, %0 acc : %0 : i32, i32, i32 -> i32
// CHECK-NEXT-GENERIC:   %3 = "kernel.mac"(%0, %0, %0) : (i32, i32, i32) -> i32

%4 = "kernel.qmac" (%0, %0, %0, %0, %0) : (i32, i32, i32, i32, i32) -> i32
// CHECK-NEXT:   %4 = kernel.qmac %0, %0 acc : %0 zp_lhs : %0 zp_rhs : %0 : i32, i32, i32, i32, i32 -> i32
// CHECK-NEXT-GENERIC:   %4 = "kernel.qmac"(%0, %0, %0, %0, %0) : (i32, i32, i32, i32, i32) -> i32


// CHECK-NEXT: }
// CHECK-NEXT-GENERIC: }) : () -> ()
