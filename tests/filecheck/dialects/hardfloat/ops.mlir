// RUN: XDSL_ROUNDTRIP

func.func @test_mul(%a : i33, %b : i33, %c : i32) -> (i33, i33, i32, i33, i33) {
  %mul = hardfloat.mul_rec_fn<24, 8>(%a, %b) : (i33, i33) -> i33
  %add = hardfloat.add_rec_fn<24, 8>(%a, %b) : (i33, i33) -> i33
  %recoded = hardfloat.fn_to_rec_fn<24, 8>(%c) : (i32) -> i33
  %unrecoded = hardfloat.rec_fn_to_fn<24, 8>(%recoded) : (i33) -> i32
  %to_int = hardfloat.rec_fn_to_in<24, 8, 32>(%recoded) <{signedness = #builtin.signedness<unsigned>}> : (i33) -> i32
  %int_to_rec = hardfloat.in_to_rec_fn<24, 8, 32>(%to_int) <{signedness = #builtin.signedness<unsigned>}> : (i32) -> i33
  %to_sint = hardfloat.rec_fn_to_in<24, 8, 32>(%recoded) <{signedness = #builtin.signedness<signed>}> : (i33) -> i32
  %sint_to_rec = hardfloat.in_to_rec_fn<24, 8, 32>(%to_sint) <{signedness = #builtin.signedness<signed>}> : (i32) -> i33
  func.return %mul, %add, %unrecoded, %int_to_rec, %sint_to_rec: i33, i33, i32, i33, i33
}

// CHECK: func.func @test_mul(%a : i33, %b : i33, %c : i32) -> (i33, i33, i32, i33, i33) {
// CHECK-NEXT:   %mul = hardfloat.mul_rec_fn<24, 8>(%a, %b) : (i33, i33) -> i33
// CHECK-NEXT:   %add = hardfloat.add_rec_fn<24, 8>(%a, %b) : (i33, i33) -> i33
// CHECK-NEXT:   %recoded = hardfloat.fn_to_rec_fn<24, 8>(%c) : (i32) -> i33
// CHECK-NEXT:   %unrecoded = hardfloat.rec_fn_to_fn<24, 8>(%recoded) : (i33) -> i32
// CHECK-NEXT:   %to_int = hardfloat.rec_fn_to_in<24, 8, 32>(%recoded) <{signedness = #builtin.signedness<unsigned>}> : (i33) -> i32
// CHECK-NEXT:   %int_to_rec = hardfloat.in_to_rec_fn<24, 8, 32>(%to_int) <{signedness = #builtin.signedness<unsigned>}> : (i32) -> i33
// CHECK-NEXT:   %to_sint = hardfloat.rec_fn_to_in<24, 8, 32>(%recoded) <{signedness = #builtin.signedness<signed>}> : (i33) -> i32
// CHECK-NEXT:   %sint_to_rec = hardfloat.in_to_rec_fn<24, 8, 32>(%to_sint) <{signedness = #builtin.signedness<signed>}> : (i32) -> i33
// CHECK-NEXT:   func.return %mul, %add, %unrecoded, %int_to_rec, %sint_to_rec : i33, i33, i32, i33, i33
// CHECK-NEXT: }
