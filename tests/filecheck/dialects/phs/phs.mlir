// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_SINGLETRIP

phs.pe @myfirstaccelerator with %0, %1, %2, %3, %4, %5 (%6 : f32, %7 : f32) {
  %8 = phs.choose @_0 with %0 (%6 : f32, %7 : f32) -> f32
    0) (%9, %10) {
      %11 = arith.mulf %9, %10 : f32
      phs.yield %11 : f32
    }
    1) (%12, %13) {
      %14 = arith.addf %12, %13 : f32
      phs.yield %14 : f32
    }
    2) (%15, %16) {
      %17 = arith.subf %15, %16 : f32
      phs.yield %17 : f32
    }
  %9 = phs.choose @_1 with %1 (%6 : f32, %8 : f32) -> f32
    0) (%10, %11) {
      %12 = arith.mulf %10, %11 : f32
      phs.yield %12 : f32
    }
    1) (%13, %14) {
      %15 = arith.addf %13, %14 : f32
      phs.yield %15 : f32
    }
  %10 = phs.mux with %2 (%8 : f32, %9 : f32) -> f32
  %11 = phs.mux with %5 (%9 : f32, %7 : f32) -> f32
  %12 = phs.choose @_2 with %3 (%8 : f32, %11 : f32) -> f32
    0) (%13, %14) {
      %15 = arith.mulf %13, %14 : f32
      phs.yield %15 : f32
    }
    1) (%16, %17) {
      %18 = arith.divf %16, %17 : f32
      phs.yield %18 : f32
    }
  %13 = phs.mux with %4 (%10 : f32, %12 : f32) -> f32
  phs.yield %13 : f32
}


phs.pe @myfirstswitchlessaccelerator (%0 : f32, %1 : f32) {
  %2 = arith.mulf %0, %1 : f32
  phs.yield %2 : f32
}

// CHECK: builtin.module {
// CHECK-NEXT:   phs.pe @myfirstaccelerator with %0, %1, %2, %3, %4, %5 (%6 : f32, %7 : f32) {
// CHECK-NEXT:     %8 = phs.choose @_0 with %0 (%6 : f32, %7 : f32) -> f32
// CHECK-NEXT:       0) (%9, %10) {
// CHECK-NEXT:         %11 = arith.mulf %9, %10 : f32
// CHECK-NEXT:         phs.yield %11 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:       1) (%12, %13) {
// CHECK-NEXT:         %14 = arith.addf %12, %13 : f32
// CHECK-NEXT:         phs.yield %14 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:       2) (%15, %16) {
// CHECK-NEXT:         %17 = arith.subf %15, %16 : f32
// CHECK-NEXT:         phs.yield %17 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:     %9 = phs.choose @_1 with %1 (%6 : f32, %8 : f32) -> f32
// CHECK-NEXT:       0) (%10, %11) {
// CHECK-NEXT:         %12 = arith.mulf %10, %11 : f32
// CHECK-NEXT:         phs.yield %12 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:       1) (%13, %14) {
// CHECK-NEXT:         %15 = arith.addf %13, %14 : f32
// CHECK-NEXT:         phs.yield %15 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:     %10 = phs.mux with %2 (%8 : f32, %9 : f32) -> f32
// CHECK-NEXT:     %11 = phs.mux with %5 (%9 : f32, %7 : f32) -> f32
// CHECK-NEXT:     %12 = phs.choose @_2 with %3 (%8 : f32, %11 : f32) -> f32
// CHECK-NEXT:       0) (%13, %14) {
// CHECK-NEXT:         %15 = arith.mulf %13, %14 : f32
// CHECK-NEXT:         phs.yield %15 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:       1) (%16, %17) {
// CHECK-NEXT:         %18 = arith.divf %16, %17 : f32
// CHECK-NEXT:         phs.yield %18 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:     %13 = phs.mux with %4 (%10 : f32, %12 : f32) -> f32
// CHECK-NEXT:     phs.yield %13 : f32
// CHECK-NEXT:   }
// CHECK-NEXT:   phs.pe @myfirstswitchlessaccelerator (%0 : f32, %1 : f32) {
// CHECK-NEXT:     %2 = arith.mulf %0, %1 : f32
// CHECK-NEXT:     phs.yield %2 : f32
// CHECK-NEXT:   }
// CHECK-NEXT: }
