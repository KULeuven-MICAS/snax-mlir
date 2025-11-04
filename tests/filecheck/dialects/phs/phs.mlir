// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_SINGLETRIP

builtin.module {
  phs.pe @myfirstaccelerator with %0, %1, %2, %3, %4, %5 (%6 : f32, %7 : f32) {
    %8 = phs.choose @_0 with %0 (%6 : f32, %7 : f32) -> f32 {
      0) arith.mulf
      1) arith.addf
    }
    %9 = phs.choose @_1 with %1 (%6 : f32, %8 : f32) -> f32 {
      0) arith.mulf
      1) arith.addf
    }
    %10 = phs.mux with %2 (%8 : f32, %9 : f32) -> f32
    %11 = phs.mux with %5 (%9 : f32, %7 : f32) -> f32
    %12 = phs.choose @_2 with %3 (%8 : f32, %11 : f32) -> f32 {
      0) arith.mulf
      1) arith.divf
    }
    %13 = phs.mux with %4 (%10 : f32, %12 : f32) -> f32
    phs.yield %13 : f32
  }
}

// CHECK: builtin.module {
// CHECK-NEXT:   phs.pe @myfirstaccelerator with %0, %1, %2, %3, %4, %5 (%6 : f32, %7 : f32) {
// CHECK-NEXT:     %8 = phs.choose @_0 with %0 (%6 : f32, %7 : f32) -> f32 {
// CHECK-NEXT:       0) arith.mulf
// CHECK-NEXT:       1) arith.addf
// CHECK-NEXT:     }
// CHECK-NEXT:     %9 = phs.choose @_1 with %1 (%6 : f32, %8 : f32) -> f32 {
// CHECK-NEXT:       0) arith.mulf
// CHECK-NEXT:       1) arith.addf
// CHECK-NEXT:     }
// CHECK-NEXT:     %10 = phs.mux with %2 (%8 : f32, %9 : f32) -> f32
// CHECK-NEXT:     %11 = phs.mux with %5 (%9 : f32, %7 : f32) -> f32
// CHECK-NEXT:     %12 = phs.choose @_2 with %3 (%8 : f32, %11 : f32) -> f32 {
// CHECK-NEXT:       0) arith.mulf
// CHECK-NEXT:       1) arith.divf
// CHECK-NEXT:     }
// CHECK-NEXT:     %13 = phs.mux with %4 (%10 : f32, %12 : f32) -> f32
// CHECK-NEXT:     phs.yield %13 : f32
// CHECK-NEXT:   }
// CHECK-NEXT: }
