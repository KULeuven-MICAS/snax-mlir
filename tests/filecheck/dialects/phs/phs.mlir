// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_SINGLETRIP

"phs.pe"() <{sym_name = "myfirstaccelerator", function_type = (f32, f32, index, index, index, index, index, index) -> f32}> ({
^bb0(%0 : f32, %1 : f32, %2 : index, %3 : index, %4 : index, %5 : index, %6 : index, %7 : index):
  %8 = phs.choose @_0 with %2 (%0 : f32, %1 : f32) -> f32 {
    0) arith.mulf
    1) arith.addf
  }
  %9 = phs.choose @_1 with %3 (%0 : f32, %8 : f32) -> f32 {
    0) arith.mulf
    1) arith.addf
  }
  %10 = phs.mux(%8 : f32, %9 : f32) -> f32 with %4
  %11 = phs.mux(%9 : f32, %1 : f32) -> f32 with %7
  %12 = phs.choose @_2 with %5 (%8 : f32, %11 : f32) -> f32 {
    0) arith.mulf
    1) arith.divf
  }
  %13 = phs.mux(%10 : f32, %12 : f32) -> f32 with %6
  phs.yield %13 : f32
}) : () -> ()


// CHECK: "phs.pe"() <{sym_name = "myfirstaccelerator", function_type = (f32, f32, index, index, index, index, index, index) -> f32}> ({
// CHECK-NEXT: ^bb0(%0 : f32, %1 : f32, %2 : index, %3 : index, %4 : index, %5 : index, %6 : index, %7 : index):
// CHECK-NEXT:   %8 = phs.choose @_0 with %2 (%0 : f32, %1 : f32) -> f32 {
// CHECK-NEXT:     0) arith.mulf
// CHECK-NEXT:     1) arith.addf
// CHECK-NEXT:   }
// CHECK-NEXT:   %9 = phs.choose @_1 with %3 (%0 : f32, %8 : f32) -> f32 {
// CHECK-NEXT:     0) arith.mulf
// CHECK-NEXT:     1) arith.addf
// CHECK-NEXT:   }
// CHECK-NEXT:   %10 = phs.mux(%8 : f32, %9 : f32) -> f32 with %4
// CHECK-NEXT:   %11 = phs.mux(%9 : f32, %1 : f32) -> f32 with %7
// CHECK-NEXT:   %12 = phs.choose @_2 with %5 (%8 : f32, %11 : f32) -> f32 {
// CHECK-NEXT:     0) arith.mulf
// CHECK-NEXT:     1) arith.divf
// CHECK-NEXT:   }
// CHECK-NEXT:   %13 = phs.mux(%10 : f32, %12 : f32) -> f32 with %6
// CHECK-NEXT:   phs.yield %13 : f32
// CHECK-NEXT: }) : () -> ()
