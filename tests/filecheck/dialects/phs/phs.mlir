// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_SINGLETRIP

phs.pe @myfirstaccelerator with %0, %1, %2, %3, %4, %5 (%6 : f32, %7 : f32) {
  %8 = phs.choose @_0 with %0 (%6 : f32, %7 : f32) -> f32
    0) {
      %9 = arith.mulf %6, %7 : f32
      phs.yield %9 : f32
    }
    1) {
      %10 = arith.addf %6, %7 : f32
      phs.yield %10 : f32
    }
    2) {
      %11 = arith.subf %6, %7 : f32
      phs.yield %11 : f32
    }
  %12 = phs.choose @_1 with %1 (%6 : f32, %8 : f32) -> f32
    0) {
      %13 = arith.mulf %6, %8 : f32
      phs.yield %13 : f32
    }
    1) {
      %14 = arith.addf %6, %8 : f32
      phs.yield %14 : f32
    }
  %15 = phs.mux with %2 (%8 : f32, %12 : f32) -> f32
  %16 = phs.mux with %5 (%12 : f32, %7 : f32) -> f32
  %17 = phs.choose @_2 with %3 (%8 : f32, %16 : f32) -> f32
    0) {
      %18 = arith.mulf %8, %12 : f32
      phs.yield %18 : f32
    }
    1) {
      %19 = arith.divf %8, %16 : f32
      phs.yield %19 : f32
    }
  %20 = phs.mux with %4 (%15 : f32, %17 : f32) -> f32
  phs.yield %20 : f32
}

phs.pe @myfirstswitchslessaccelerator (%0 : f32, %1 : f32) {
  %2 = arith.mulf %0, %1 : f32
  phs.yield %2 : f32
}


// CHECK: builtin.module {
// CHECK-NEXT:   phs.pe @myfirstaccelerator with %0, %1, %2, %3, %4, %5 (%6 : f32, %7 : f32) {
// CHECK-NEXT:     %8 = phs.choose @_0 with %0 (%6 : f32, %7 : f32) -> f32
// CHECK-NEXT:       0) {
// CHECK-NEXT:         %9 = arith.mulf %6, %7 : f32
// CHECK-NEXT:         phs.yield %9 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:       1) {
// CHECK-NEXT:         %10 = arith.addf %6, %7 : f32
// CHECK-NEXT:         phs.yield %10 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:       2) {
// CHECK-NEXT:         %11 = arith.subf %6, %7 : f32
// CHECK-NEXT:         phs.yield %11 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:     %12 = phs.choose @_1 with %1 (%6 : f32, %8 : f32) -> f32
// CHECK-NEXT:       0) {
// CHECK-NEXT:         %13 = arith.mulf %6, %8 : f32
// CHECK-NEXT:         phs.yield %13 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:       1) {
// CHECK-NEXT:         %14 = arith.addf %6, %8 : f32
// CHECK-NEXT:         phs.yield %14 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:     %15 = phs.mux with %2 (%8 : f32, %12 : f32) -> f32
// CHECK-NEXT:     %16 = phs.mux with %5 (%12 : f32, %7 : f32) -> f32
// CHECK-NEXT:     %17 = phs.choose @_2 with %3 (%8 : f32, %16 : f32) -> f32
// CHECK-NEXT:       0) {
// CHECK-NEXT:         %18 = arith.mulf %8, %12 : f32
// CHECK-NEXT:         phs.yield %18 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:       1) {
// CHECK-NEXT:         %19 = arith.divf %8, %16 : f32
// CHECK-NEXT:         phs.yield %19 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:     %20 = phs.mux with %4 (%15 : f32, %17 : f32) -> f32
// CHECK-NEXT:     phs.yield %20 : f32
// CHECK-NEXT:   }
// CHECK-NEXT:   phs.pe @myfirstswitchslessaccelerator (%0 : f32, %1 : f32) {
// CHECK-NEXT:     %2 = arith.mulf %0, %1 : f32
// CHECK-NEXT:     phs.yield %2 : f32
// CHECK-NEXT:   }
// CHECK-NEXT: }
