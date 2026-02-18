// RUN: snax-opt -p convert-float-to-hardfloat %s | filecheck %s

func.func @test_hardfloat(%a : f32, %b : f32, %c: i32) -> (f32, f32, f32, f32) {
  %add = arith.addf %a, %b : f32
  %mul = arith.mulf %a, %b : f32
  %sfp = arith.sitofp %c : i32 to f32
  %ufp = arith.uitofp %c : i32 to f32
  return %add, %mul, %sfp, %ufp : f32, f32, f32, f32
}

// CHECK: func.func @test_hardfloat(%a : f32, %b : f32, %c : i32) -> (f32, f32, f32, f32) {
// CHECK-NEXT:   %add = builtin.unrealized_conversion_cast %a : f32 to i32
// CHECK-NEXT:   %add_1 = builtin.unrealized_conversion_cast %b : f32 to i32
// CHECK-NEXT:   %add_2 = hardfloat.fn_to_rec_fn<24, 8>(%add) : (i32) -> i33
// CHECK-NEXT:   %add_3 = hardfloat.fn_to_rec_fn<24, 8>(%add_1) : (i32) -> i33
// CHECK-NEXT:   %add_4 = hardfloat.add_rec_fn<24, 8>(%add_2, %add_3) : (i33, i33) -> i33
// CHECK-NEXT:   %add_5 = hardfloat.rec_fn_to_fn<24, 8>(%add_4) : (i33) -> i32
// CHECK-NEXT:   %add_6 = builtin.unrealized_conversion_cast %add_5 : i32 to f32
// CHECK-NEXT:   %mul = builtin.unrealized_conversion_cast %a : f32 to i32
// CHECK-NEXT:   %mul_1 = builtin.unrealized_conversion_cast %b : f32 to i32
// CHECK-NEXT:   %mul_2 = hardfloat.fn_to_rec_fn<24, 8>(%mul) : (i32) -> i33
// CHECK-NEXT:   %mul_3 = hardfloat.fn_to_rec_fn<24, 8>(%mul_1) : (i32) -> i33
// CHECK-NEXT:   %mul_4 = hardfloat.mul_rec_fn<24, 8>(%mul_2, %mul_3) : (i33, i33) -> i33
// CHECK-NEXT:   %mul_5 = hardfloat.rec_fn_to_fn<24, 8>(%mul_4) : (i33) -> i32
// CHECK-NEXT:   %mul_6 = builtin.unrealized_conversion_cast %mul_5 : i32 to f32
// CHECK-NEXT:   %sfp = hardfloat.in_to_rec_fn<24, 8, 32>(%c) <{signedness = #builtin.signedness<signed>}> : (i32) -> i33
// CHECK-NEXT:   %sfp_1 = hardfloat.rec_fn_to_fn<24, 8>(%sfp) : (i33) -> i32
// CHECK-NEXT:   %sfp_2 = builtin.unrealized_conversion_cast %sfp_1 : i32 to f32
// CHECK-NEXT:   %ufp = hardfloat.in_to_rec_fn<24, 8, 32>(%c) <{signedness = #builtin.signedness<unsigned>}> : (i32) -> i33
// CHECK-NEXT:   %ufp_1 = hardfloat.rec_fn_to_fn<24, 8>(%ufp) : (i33) -> i32
// CHECK-NEXT:   %ufp_2 = builtin.unrealized_conversion_cast %ufp_1 : i32 to f32
// CHECK-NEXT:   func.return %add_6, %mul_6, %sfp_2, %ufp_2 : f32, f32, f32, f32
// CHECK-NEXT: }
