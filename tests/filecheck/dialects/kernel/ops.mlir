// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK: builtin.module {
// CHECK-GENERIC: "builtin.module"() ({

%0 = "test.op"() : () -> i32
// CHECK-NEXT:   %0 = "test.op"() : () -> i32
// CHECK-GENERIC-NEXT:   %0 = "test.op"() : () -> i32

%1 = "kernel.add" (%0, %0) : (i32, i32) -> i32
// CHECK-NEXT:   %1 = kernel.add %0, %0 : i32, i32 -> i32
// CHECK-GENERIC-NEXT:   %1 = "kernel.add"(%0, %0) : (i32, i32) -> i32

%2 = "kernel.mul" (%0, %0) : (i32, i32) -> i32
// CHECK-NEXT:   %2 = kernel.mul %0, %0 : i32, i32 -> i32
// CHECK-GENERIC-NEXT:   %2 = "kernel.mul"(%0, %0) : (i32, i32) -> i32

%3 = "kernel.mac" (%0, %0) : (i32, i32) -> i32
// CHECK-NEXT:   %3 = kernel.mac %0, %0 : i32, i32 -> i32
// CHECK-GENERIC-NEXT:   %3 = "kernel.mac"(%0, %0) : (i32, i32) -> i32

%4 = "kernel.qmac" (%0, %0, %0, %0) : (i32, i32, i32, i32) -> i32
// CHECK-NEXT:   %4 = kernel.qmac %0, %0 zp_lhs : %0 zp_rhs : %0 : i32, i32, i32, i32 -> i32
// CHECK-GENERIC-NEXT:   %4 = "kernel.qmac"(%0, %0, %0, %0) : (i32, i32, i32, i32) -> i32

%5 = "kernel.rescale"(%0) {input_zp = 23 : i8, output_zp = -23 : i8, multiplier = array<i32: 12345>, shift = array<i8: 30>, min_int = -128 : i8, max_int = 127 : i8, double_round = true} : (i32) -> i32
// CHECK-NEXT:   %5 = kernel.rescale %0 {input_zp = 23 : i8, output_zp = -23 : i8, multiplier = array<i32: 12345>, shift = array<i8: 30>, min_int = -128 : i8, max_int = 127 : i8, double_round = true} : (i32) -> i32
// CHECK-GENERIC-NEXT    %5 = "kernel.rescale"(%0) {input_zp = 23 : i8, output_zp = -23 : i8, multiplier = array<i32: 12345>, shift = array<i8: 30>, min_int = -128 : i8, max_int = 127 : i8, double_round = true} : (i32) -> i32
