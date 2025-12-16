// RUN: snax-opt %s -p convert-pe-to-hw | filecheck %s

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
// CHECK-NEXT:   hw.module private @myfirstaccelerator(in %0 data_0: i32, in %1 data_1: i32, in %2 switch_0: i2, in %3 switch_1: i1, in %4 switch_2: i1, in %5 switch_3: i1, in %6 switch_4: i1, in %7 switch_5: i1, out out_0: i32) {
// CHECK-NEXT:     %8 = builtin.unrealized_conversion_cast %2 : i2 to index
// CHECK-NEXT:     %9 = builtin.unrealized_conversion_cast %3 : i1 to index
// CHECK-NEXT:     %10 = builtin.unrealized_conversion_cast %4 : i1 to index
// CHECK-NEXT:     %11 = builtin.unrealized_conversion_cast %5 : i1 to index
// CHECK-NEXT:     %12 = builtin.unrealized_conversion_cast %6 : i1 to index
// CHECK-NEXT:     %13 = builtin.unrealized_conversion_cast %7 : i1 to index
// CHECK-NEXT:     %14 = phs.choose @_0 with %8 (%0 : i32, %1 : i32) -> i32
// CHECK-NEXT:       0) {
// CHECK-NEXT:         %15 = arith.muli %0, %1 : i32
// CHECK-NEXT:         phs.yield %15 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       1) {
// CHECK-NEXT:         %16 = arith.addi %0, %1 : i32
// CHECK-NEXT:         phs.yield %16 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       2) {
// CHECK-NEXT:         %17 = arith.subi %0, %1 : i32
// CHECK-NEXT:         phs.yield %17 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:     %18 = phs.choose @_1 with %9 (%0 : i32, %14 : i32) -> i32
// CHECK-NEXT:       0) {
// CHECK-NEXT:         %19 = arith.muli %0, %14 : i32
// CHECK-NEXT:         phs.yield %19 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       1) {
// CHECK-NEXT:         %20 = arith.addi %0, %14 : i32
// CHECK-NEXT:         phs.yield %20 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:     %21 = phs.mux with %10 (%14 : i32, %18 : i32) -> i32
// CHECK-NEXT:     %22 = phs.mux with %13 (%18 : i32, %1 : i32) -> i32
// CHECK-NEXT:     %23 = phs.choose @_2 with %11 (%14 : i32, %22 : i32) -> i32
// CHECK-NEXT:       0) {
// CHECK-NEXT:         %24 = arith.muli %14, %18 : i32
// CHECK-NEXT:         phs.yield %24 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       1) {
// CHECK-NEXT:         %25 = arith.divui %14, %22 : i32
// CHECK-NEXT:         phs.yield %25 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:     %26 = phs.mux with %12 (%21 : i32, %23 : i32) -> i32
// CHECK-NEXT:     hw.output %26 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }
