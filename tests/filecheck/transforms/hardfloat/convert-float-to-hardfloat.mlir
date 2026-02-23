// RUN: snax-opt -p convert-float-to-hardfloat %s | filecheck %s

func.func @test_hardfloat(%a : f32, %b : f32, %c: i32, %d : f32) -> (f32, f32, f32, f32, i32, i32) {
  %add = arith.addf %a, %b : f32
  %mul = arith.mulf %a, %b : f32
  %sfp = arith.sitofp %c : i32 to f32
  %ufp = arith.uitofp %c : i32 to f32
  %sint = arith.fptosi %d : f32 to i32
  %uint = arith.fptoui %d : f32 to i32
  return %add, %mul, %sfp, %ufp, %uint, %sint : f32, f32, f32, f32, i32, i32
}

// CHECK: func.func @test_hardfloat(%a : f32, %b : f32, %c : i32, %d : f32) -> (f32, f32, f32, f32, i32, i32) {
// CHECK-NEXT:   %add = hw.constant false
// CHECK-NEXT:   %add_1 = builtin.unrealized_conversion_cast %a : f32 to i32
// CHECK-NEXT:   %add_2 = builtin.unrealized_conversion_cast %b : f32 to i32
// CHECK-NEXT:   %add_3 = hardfloat.fn_to_rec_fn<24, 8>(%add_1) : (i32) -> i33
// CHECK-NEXT:   %add_4 = hardfloat.fn_to_rec_fn<24, 8>(%add_2) : (i32) -> i33
// CHECK-NEXT:   %add_5 = hw.constant 0 : i3
// CHECK-NEXT:   %add_6 = hw.constant true
// CHECK-NEXT:   %add_7, %add_8 = hardfloat.add_rec_fn<24, 8>(%add, %add_3, %add_4, %add_5, %add_6) : (i1, i33, i33, i3, i1) -> (i33, i5)
// CHECK-NEXT:   %add_9 = hardfloat.rec_fn_to_fn<24, 8>(%add_7) : (i33) -> i32
// CHECK-NEXT:   %add_10 = builtin.unrealized_conversion_cast %add_9 : i32 to f32
// CHECK-NEXT:   %mul = builtin.unrealized_conversion_cast %a : f32 to i32
// CHECK-NEXT:   %mul_1 = builtin.unrealized_conversion_cast %b : f32 to i32
// CHECK-NEXT:   %mul_2 = hardfloat.fn_to_rec_fn<24, 8>(%mul) : (i32) -> i33
// CHECK-NEXT:   %mul_3 = hardfloat.fn_to_rec_fn<24, 8>(%mul_1) : (i32) -> i33
// CHECK-NEXT:   %mul_4 = hw.constant 0 : i3
// CHECK-NEXT:   %mul_5 = hw.constant true
// CHECK-NEXT:   %mul_6, %mul_7 = hardfloat.mul_rec_fn<24, 8>(%mul_2, %mul_3, %mul_4, %mul_5) : (i33, i33, i3, i1) -> (i33, i5)
// CHECK-NEXT:   %mul_8 = hardfloat.rec_fn_to_fn<24, 8>(%mul_6) : (i33) -> i32
// CHECK-NEXT:   %mul_9 = builtin.unrealized_conversion_cast %mul_8 : i32 to f32
// CHECK-NEXT:   %sfp = hw.constant true
// CHECK-NEXT:   %sfp_1 = hw.constant 0 : i3
// CHECK-NEXT:   %sfp_2 = hw.constant true
// CHECK-NEXT:   %sfp_3, %sfp_4 = hardfloat.in_to_rec_fn<24, 8, 32>(%sfp, %c, %sfp_1, %sfp_2) : (i1, i32, i3, i1) -> (i33, i5)
// CHECK-NEXT:   %sfp_5 = hardfloat.rec_fn_to_fn<24, 8>(%sfp_3) : (i33) -> i32
// CHECK-NEXT:   %sfp_6 = builtin.unrealized_conversion_cast %sfp_5 : i32 to f32
// CHECK-NEXT:   %ufp = hw.constant false
// CHECK-NEXT:   %ufp_1 = hw.constant 0 : i3
// CHECK-NEXT:   %ufp_2 = hw.constant true
// CHECK-NEXT:   %ufp_3, %ufp_4 = hardfloat.in_to_rec_fn<24, 8, 32>(%ufp, %c, %ufp_1, %ufp_2) : (i1, i32, i3, i1) -> (i33, i5)
// CHECK-NEXT:   %ufp_5 = hardfloat.rec_fn_to_fn<24, 8>(%ufp_3) : (i33) -> i32
// CHECK-NEXT:   %ufp_6 = builtin.unrealized_conversion_cast %ufp_5 : i32 to f32
// CHECK-NEXT:   %sint = hw.constant true
// CHECK-NEXT:   %sint_1 = hw.constant 0 : i3
// CHECK-NEXT:   %sint_2 = builtin.unrealized_conversion_cast %d : f32 to i32
// CHECK-NEXT:   %sint_3 = hardfloat.fn_to_rec_fn<24, 8>(%sint_2) : (i32) -> i33
// CHECK-NEXT:   %sint_4, %sint_5 = hardfloat.rec_fn_to_in<24, 8, 32>(%sint_3, %sint_1, %sint) : (i33, i3, i1) -> (i32, i3)
// CHECK-NEXT:   %uint = hw.constant false
// CHECK-NEXT:   %uint_1 = hw.constant 0 : i3
// CHECK-NEXT:   %uint_2 = builtin.unrealized_conversion_cast %d : f32 to i32
// CHECK-NEXT:   %uint_3 = hardfloat.fn_to_rec_fn<24, 8>(%uint_2) : (i32) -> i33
// CHECK-NEXT:   %uint_4, %uint_5 = hardfloat.rec_fn_to_in<24, 8, 32>(%uint_3, %uint_1, %uint) : (i33, i3, i1) -> (i32, i3)
// CHECK-NEXT:   func.return %add_10, %mul_9, %sfp_6, %ufp_6, %uint_4, %sint_4 : f32, f32, f32, f32, i32, i32
// CHECK-NEXT: }
