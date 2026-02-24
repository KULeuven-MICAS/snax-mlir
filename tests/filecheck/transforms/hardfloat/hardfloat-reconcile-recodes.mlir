// RUN: snax-opt -p hardfloat-reconcile-recodes %s | filecheck %s


func.func @test_hardfloat(%a : f32, %b : f32) -> f32 {
  %false = arith.constant 0 : i1
  %rm = arith.constant 0 : i3
  %a_i32 = builtin.unrealized_conversion_cast %a : f32 to i32
  %b_i32 = builtin.unrealized_conversion_cast %b : f32 to i32
  %a_rec = hardfloat.fn_to_rec_fn<24, 8>(%a_i32) : (i32) -> i33
  %b_rec = hardfloat.fn_to_rec_fn<24, 8>(%b_i32) : (i32) -> i33
  %add_res, %add_flags = hardfloat.add_rec_fn<24, 8>(%false, %a_rec, %b_rec, %rm, %false) : (i1, i33, i33, i3, i1) -> (i33, i5)
  %add_val_i32 = hardfloat.rec_fn_to_fn<24, 8>(%add_res) : (i33) -> i32
  %add_f32 = builtin.unrealized_conversion_cast %add_val_i32 : i32 to f32
  %mul_lhs_i32 = builtin.unrealized_conversion_cast %add_f32 : f32 to i32
  %mul_lhs_rec = hardfloat.fn_to_rec_fn<24, 8>(%mul_lhs_i32) : (i32) -> i33
  %mul_res, %mul_flags = hardfloat.mul_rec_fn<24, 8>(%mul_lhs_rec, %b_rec, %rm, %false) : (i33, i33, i3, i1) -> (i33, i5)
  %mul_val_i32 = hardfloat.rec_fn_to_fn<24, 8>(%mul_res) : (i33) -> i32
  %final_f32 = builtin.unrealized_conversion_cast %mul_val_i32 : i32 to f32
  func.return %final_f32 : f32
}


// CHECK: func.func @test_hardfloat(%a : f32, %b : f32) -> f32 {
// CHECK-NEXT:   %false = arith.constant false
// CHECK-NEXT:   %rm = arith.constant 0 : i3
// CHECK-NEXT:   %a_i32 = builtin.unrealized_conversion_cast %a : f32 to i32
// CHECK-NEXT:   %b_i32 = builtin.unrealized_conversion_cast %b : f32 to i32
// CHECK-NEXT:   %a_rec = hardfloat.fn_to_rec_fn<24, 8>(%a_i32) : (i32) -> i33
// CHECK-NEXT:   %b_rec = hardfloat.fn_to_rec_fn<24, 8>(%b_i32) : (i32) -> i33
// CHECK-NEXT:   %add_res, %add_flags = hardfloat.add_rec_fn<24, 8>(%false, %a_rec, %b_rec, %rm, %false) : (i1, i33, i33, i3, i1) -> (i33, i5)
// CHECK-NEXT:   %mul_res, %mul_flags = hardfloat.mul_rec_fn<24, 8>(%add_res, %b_rec, %rm, %false) : (i33, i33, i3, i1) -> (i33, i5)
// CHECK-NEXT:   %mul_val_i32 = hardfloat.rec_fn_to_fn<24, 8>(%mul_res) : (i33) -> i32
// CHECK-NEXT:   %final_f32 = builtin.unrealized_conversion_cast %mul_val_i32 : i32 to f32
// CHECK-NEXT:   func.return %final_f32 : f32
// CHECK-NEXT: }
