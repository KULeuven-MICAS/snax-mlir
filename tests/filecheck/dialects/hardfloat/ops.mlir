// RUN: XDSL_ROUNDTRIP


func.func @test_hardfloat_ops(%a : i33, %b : i33, %float_in : i32) -> (i33, i33, i32, i33, i33) {
  // Control signals required by the new operand definitions
  %false = arith.constant 0 : i1
  %true = arith.constant 1 : i1
  %rm = arith.constant 0 : i3

  // %mul: (a, b, roundingMode, detectTininess)
  %mul = hardfloat.mul_rec_fn<24, 8>(%a, %b, %rm, %false) : (i33, i33, i3, i1) -> i33

  // %add: (subOp, a, b, roundingMode, detectTininess)
  %add = hardfloat.add_rec_fn<24, 8>(%false, %a, %b, %rm, %false) : (i1, i33, i33, i3, i1) -> i33

  // %recoded: (input)
  %recoded = hardfloat.fn_to_rec_fn<24, 8>(%float_in) : (i32) -> i33

  // %unrecoded: (input)
  %unrecoded = hardfloat.rec_fn_to_fn<24, 8>(%recoded) : (i33) -> i32

  // %to_int: (input, signedOut)
  %to_int = hardfloat.rec_fn_to_in<24, 8, 32>(%recoded, %false) : (i33, i1) -> i32

  // %int_to_rec: (signedIn, input)
  %int_to_rec = hardfloat.in_to_rec_fn<24, 8, 32>(%false, %to_int) : (i1, i32) -> i33

  // %to_sint: (input, signedOut)
  %to_sint = hardfloat.rec_fn_to_in<24, 8, 32>(%recoded, %true) : (i33, i1) -> i32

  // %sint_to_rec: (signedIn, input)
  %sint_to_rec = hardfloat.in_to_rec_fn<24, 8, 32>(%true, %to_sint) : (i1, i32) -> i33

  func.return %mul, %add, %unrecoded, %int_to_rec, %sint_to_rec : i33, i33, i32, i33, i33
}

// CHECK:       func.func @test_hardfloat_ops(%{{.*}} : i33, %{{.*}} : i33, %{{.*}} : i32) -> (i33, i33, i32, i33, i33) {
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    %rm = arith.constant 0 : i3
// CHECK-NEXT:    %mul = hardfloat.mul_rec_fn<24, 8>(%a, %b, %rm, %false) : (i33, i33, i3, i1) -> i33
// CHECK-NEXT:    %add = hardfloat.add_rec_fn<24, 8>(%false, %a, %b, %rm, %false) : (i1, i33, i33, i3, i1) -> i33
// CHECK-NEXT:    %recoded = hardfloat.fn_to_rec_fn<24, 8>(%float_in) : (i32) -> i33
// CHECK-NEXT:    %unrecoded = hardfloat.rec_fn_to_fn<24, 8>(%recoded) : (i33) -> i32
// CHECK-NEXT:    %to_int = hardfloat.rec_fn_to_in<24, 8, 32>(%recoded, %false) : (i33, i1) -> i32
// CHECK-NEXT:    %int_to_rec = hardfloat.in_to_rec_fn<24, 8, 32>(%false, %to_int) : (i1, i32) -> i33
// CHECK-NEXT:    %to_sint = hardfloat.rec_fn_to_in<24, 8, 32>(%recoded, %true) : (i33, i1) -> i32
// CHECK-NEXT:    %sint_to_rec = hardfloat.in_to_rec_fn<24, 8, 32>(%true, %to_sint) : (i1, i32) -> i33
// CHECK-NEXT:    func.return %mul, %add, %unrecoded, %int_to_rec, %sint_to_rec : i33, i33, i32, i33, i33
// CHECK-NEXT:  }
