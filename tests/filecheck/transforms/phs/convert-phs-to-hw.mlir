// RUN: snax-opt %s -p convert-phs-to-hw | filecheck %s

phs.pe @myfirstaccelerator with %0, %1, %2, %3, %4, %5 (%6 : i32, %7 : i32) {
  %8 = phs.choose @_0 with %0 (%6 : i32, %7 : i32) -> i32
    0) {
      %9 = arith.muli %6, %7 : i32
      phs.yield %9 : i32
    }
    1) {
      %10 = arith.addi %6, %7 : i32
      phs.yield %10 : i32
    }
    2) {
      %11 = arith.subi %6, %7 : i32
      phs.yield %11 : i32
    }
  %12 = phs.choose @_1 with %1 (%6 : i32, %8 : i32) -> i32
    0) {
      %13 = arith.muli %6, %8 : i32
      phs.yield %13 : i32
    }
    1) {
      %14 = arith.addi %6, %8 : i32
      phs.yield %14 : i32
    }
  %15 = phs.mux with %2 (%8 : i32, %12 : i32) -> i32
  %16 = phs.mux with %5 (%12 : i32, %7 : i32) -> i32
  %17 = phs.choose @_2 with %3 (%8 : i32, %16 : i32) -> i32
    0) {
      %18 = arith.muli %8, %12 : i32
      phs.yield %18 : i32
    }
    1) {
      %19 = arith.divui %8, %16 : i32
      phs.yield %19 : i32
    }
  %20 = phs.mux with %4 (%15 : i32, %17 : i32) -> i32
  phs.yield %20 : i32
}

// CHECK: builtin.module {
// CHECK-NEXT: hw.module @myfirstaccelerator(in %arg0 data_0: i32, in %arg1 data_1: i32, in %arg2 switch_0: i2, in %arg3 switch_1: i1, in %arg4 switch_2: i1, in %arg5 switch_3: i1, in %arg6 switch_4: i1, in %arg7 switch_5: i1, out out_0: i32) {
// CHECK-NEXT:     %0 = comb.mul %arg0, %arg1 : i32
// CHECK-NEXT:     %1 = comb.add %arg0, %arg1 : i32
// CHECK-NEXT:     %2 = comb.sub %arg0, %arg1 : i32
// CHECK-NEXT:     %3 = hw.array_create %0, %1, %2 : i32
// CHECK-NEXT:     %4 = hw.array_get %3[%arg2] : !hw.array<3xi32>, i2
// CHECK-NEXT:     %5 = comb.mul %arg0, %4 : i32
// CHECK-NEXT:     %6 = comb.add %arg0, %4 : i32
// CHECK-NEXT:     %7 = hw.array_create %5, %6 : i32
// CHECK-NEXT:     %8 = hw.array_get %7[%arg3] : !hw.array<2xi32>, i1
// CHECK-NEXT:     %9 = comb.mux %arg4, %8, %4 : i32
// CHECK-NEXT:     %10 = comb.mux %arg7, %arg1, %8 : i32
// CHECK-NEXT:     %11 = comb.mul %4, %8 : i32
// CHECK-NEXT:     %12 = comb.divu %4, %10 : i32
// CHECK-NEXT:     %13 = hw.array_create %11, %12 : i32
// CHECK-NEXT:     %14 = hw.array_get %13[%arg5] : !hw.array<2xi32>, i1
// CHECK-NEXT:     %15 = comb.mux %arg6, %14, %9 : i32
// CHECK-NEXT:     hw.output %15 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }
