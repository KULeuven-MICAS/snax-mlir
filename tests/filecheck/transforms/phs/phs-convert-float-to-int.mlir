// RUN: snax-opt %s -p phs-convert-float-to-int | filecheck %s

builtin.module {
  phs.pe @acc1 with %0 (%a_it : f32, %b_it : f32) {
    %add = phs.choose @i_f32_f32_o_f32_0 with %0 (%a_it : f32, %b_it : f32) -> f32
      0) (%1, %2) {
        %add_1 = arith.addf %1, %2 : f32
        phs.yield %add_1 : f32
      }
    phs.yield %add : f32
  }
}

// CHECK: builtin.module {
// CHECK-NEXT:   phs.pe @acc1 with %0 (%a_it : i32, %b_it : i32) {
// CHECK-NEXT:     %add = phs.choose @i_f32_f32_o_f32_0 with %0 (%a_it : i32, %b_it : i32) -> i32
// CHECK-NEXT:       0) (%1, %2) {
// CHECK-NEXT:         %3 = builtin.unrealized_conversion_cast %1 : i32 to f32
// CHECK-NEXT:         %4 = builtin.unrealized_conversion_cast %2 : i32 to f32
// CHECK-NEXT:         %add_1 = arith.addf %3, %4 : f32
// CHECK-NEXT:         %add_2 = builtin.unrealized_conversion_cast %add_1 : f32 to i32
// CHECK-NEXT:         phs.yield %add_2 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:     phs.yield %add : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }
