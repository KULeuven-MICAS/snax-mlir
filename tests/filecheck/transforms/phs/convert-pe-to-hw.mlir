// RUN: snax-opt %s -p convert-pe-to-hw | filecheck %s

phs.pe @myfirstaccelerator with %0, %1, %2, %3, %4, %5 (%6 : i32, %7 : i32) {
  %8 = phs.choose @_0 with %0 (%6 : i32, %7 : i32) -> i32
    0) (%9, %10) {
      %11 = arith.muli %9, %10 : i32
      phs.yield %11 : i32
    }
    1) (%12, %13) {
      %14 = arith.addi %12, %13 : i32
      phs.yield %14 : i32
    }
    2) (%15, %16) {
      %17 = arith.subi %15, %16 : i32
      phs.yield %17 : i32
    }
  %9 = phs.choose @_1 with %1 (%6 : i32, %8 : i32) -> i32
    0) (%10, %11) {
      %12 = arith.muli %10, %11 : i32
      phs.yield %12 : i32
    }
    1) (%13, %14) {
      %15 = arith.addi %13, %14 : i32
      phs.yield %15 : i32
    }
  %10 = phs.mux with %2 (%8 : i32, %9 : i32) -> i32
  %11 = phs.mux with %5 (%9 : i32, %7 : i32) -> i32
  %12 = phs.choose @_2 with %3 (%8 : i32, %11 : i32) -> i32
    0) (%13, %14) {
      %15 = arith.muli %13, %14 : i32
      phs.yield %15 : i32
    }
    1) (%16, %17) {
      %18 = arith.divui %16, %17 : i32
      phs.yield %18 : i32
    }
  %13 = phs.mux with %4 (%10 : i32, %12 : i32) -> i32
  phs.yield %13 : i32
}


// CHECK: builtin.module {
// CHECK-NEXT: hw.module private @myfirstaccelerator(in %0 data_0: i32, in %1 data_1: i32, in %2 switch_0: i2, in %3 switch_1: i1, in %4 switch_2: i1, in %5 switch_3: i1, in %6 switch_4: i1, in %7 switch_5: i1, out out_0: i32) {
// CHECK-NEXT:    %8 = builtin.unrealized_conversion_cast %2 : i2 to index
// CHECK-NEXT:    %9 = builtin.unrealized_conversion_cast %3 : i1 to index
// CHECK-NEXT:    %10 = builtin.unrealized_conversion_cast %4 : i1 to index
// CHECK-NEXT:    %11 = builtin.unrealized_conversion_cast %5 : i1 to index
// CHECK-NEXT:    %12 = builtin.unrealized_conversion_cast %6 : i1 to index
// CHECK-NEXT:    %13 = builtin.unrealized_conversion_cast %7 : i1 to index
// CHECK-NEXT:    %14 = phs.choose @_0 with %8 (%0 : i32, %1 : i32) -> i32
// CHECK-NEXT:      0) (%15, %16) {
// CHECK-NEXT:        %17 = arith.muli %15, %16 : i32
// CHECK-NEXT:        phs.yield %17 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      1) (%18, %19) {
// CHECK-NEXT:        %20 = arith.addi %18, %19 : i32
// CHECK-NEXT:        phs.yield %20 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      2) (%21, %22) {
// CHECK-NEXT:        %23 = arith.subi %21, %22 : i32
// CHECK-NEXT:        phs.yield %23 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:    %15 = phs.choose @_1 with %9 (%0 : i32, %14 : i32) -> i32
// CHECK-NEXT:      0) (%16, %17) {
// CHECK-NEXT:        %18 = arith.muli %16, %17 : i32
// CHECK-NEXT:        phs.yield %18 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      1) (%19, %20) {
// CHECK-NEXT:        %21 = arith.addi %19, %20 : i32
// CHECK-NEXT:        phs.yield %21 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:    %16 = phs.mux with %10 (%14 : i32, %15 : i32) -> i32
// CHECK-NEXT:    %17 = phs.mux with %13 (%15 : i32, %1 : i32) -> i32
// CHECK-NEXT:    %18 = phs.choose @_2 with %11 (%14 : i32, %17 : i32) -> i32
// CHECK-NEXT:      0) (%19, %20) {
// CHECK-NEXT:        %21 = arith.muli %19, %20 : i32
// CHECK-NEXT:        phs.yield %21 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      1) (%22, %23) {
// CHECK-NEXT:        %24 = arith.divui %22, %23 : i32
// CHECK-NEXT:        phs.yield %24 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:    %19 = phs.mux with %12 (%16 : i32, %18 : i32) -> i32
// CHECK-NEXT:    hw.output %19 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:}
