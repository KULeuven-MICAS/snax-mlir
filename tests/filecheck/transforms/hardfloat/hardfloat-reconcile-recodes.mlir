// RUN: snax-opt -p convert-float-to-hardfloat,hardfloat-reconcile-recodes %s | filecheck %s

func.func @test_hardfloat(%a : f32, %b : f32) -> f32 {
  %add = builtin.unrealized_conversion_cast %a : f32 to i32
  %add_1 = builtin.unrealized_conversion_cast %b : f32 to i32
  %add_2 = hardfloat.fn_to_rec_fn<24, 8>(%add) : (i32) -> i33
  %add_3 = hardfloat.fn_to_rec_fn<24, 8>(%add_1) : (i32) -> i33
  %add_4 = hardfloat.add_rec_fn<24, 8>(%add_2, %add_3) : (i33, i33) -> i33
  %add_5 = hardfloat.rec_fn_to_fn<24, 8>(%add_4) : (i33) -> i32
  %add_6 = builtin.unrealized_conversion_cast %add_5 : i32 to f32
  %mul = builtin.unrealized_conversion_cast %add_6 : f32 to i32
  %mul_1 = builtin.unrealized_conversion_cast %b : f32 to i32
  %mul_2 = hardfloat.fn_to_rec_fn<24, 8>(%mul) : (i32) -> i33
  %mul_3 = hardfloat.fn_to_rec_fn<24, 8>(%mul_1) : (i32) -> i33
  %mul_4 = hardfloat.mul_rec_fn<24, 8>(%mul_2, %mul_3) : (i33, i33) -> i33
  %mul_5 = hardfloat.rec_fn_to_fn<24, 8>(%mul_4) : (i33) -> i32
  %mul_6 = builtin.unrealized_conversion_cast %mul_5 : i32 to f32
  func.return %mul_6 : f32
}


// CHECK: func.func @test_hardfloat(%a : f32, %b : f32) -> f32 {
// CHECK-NEXT:   %add = builtin.unrealized_conversion_cast %a : f32 to i32
// CHECK-NEXT:   %add_1 = builtin.unrealized_conversion_cast %b : f32 to i32
// CHECK-NEXT:   %add_2 = hardfloat.fn_to_rec_fn<24, 8>(%add) : (i32) -> i33
// CHECK-NEXT:   %add_3 = hardfloat.fn_to_rec_fn<24, 8>(%add_1) : (i32) -> i33
// CHECK-NEXT:   %add_4 = hardfloat.add_rec_fn<24, 8>(%add_2, %add_3) : (i33, i33) -> i33
// CHECK-NEXT:   %mul = builtin.unrealized_conversion_cast %b : f32 to i32
// CHECK-NEXT:   %mul_1 = hardfloat.fn_to_rec_fn<24, 8>(%mul) : (i32) -> i33
// CHECK-NEXT:   %mul_2 = hardfloat.mul_rec_fn<24, 8>(%add_4, %mul_1) : (i33, i33) -> i33
// CHECK-NEXT:   %mul_3 = hardfloat.rec_fn_to_fn<24, 8>(%mul_2) : (i33) -> i32
// CHECK-NEXT:   %mul_4 = builtin.unrealized_conversion_cast %mul_3 : i32 to f32
// CHECK-NEXT:   func.return %mul_4 : f32
// CHECK-NEXT: }
