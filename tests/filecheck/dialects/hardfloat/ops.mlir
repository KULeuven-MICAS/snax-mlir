// RUN: XDSL_ROUNDTRIP

func.func @test_hardfloat_roundtrip(%a : i33, %b : i33, %val_i32 : i32) {
  // Setup constants for control signals
  %false = arith.constant false
  %true = arith.constant true
  %rm = arith.constant 0 : i3

  // 1. Multiply: (a, b, rm, tininess) -> (res, flags)
  %mul, %m_flags = hardfloat.mul_rec_fn<24, 8>(%a, %b, %rm, %false) : (i33, i33, i3, i1) -> (i33, i5)

  // 2. Add: (subOp, a, b, rm, tininess) -> (res, flags)
  %add, %a_flags = hardfloat.add_rec_fn<24, 8>(%false, %a, %b, %rm, %false) : (i1, i33, i33, i3, i1) -> (i33, i5)

  // 3. Float to Recoded: (in) -> (out)
  %rec = hardfloat.fn_to_rec_fn<24, 8>(%val_i32) : (i32) -> i33

  // 4. Recoded to Float: (in) -> (out)
  %f_out = hardfloat.rec_fn_to_fn<24, 8>(%rec) : (i33) -> i32

  // 5. Integer to Recoded: (signedIn, in, rm, tininess) -> (out, flags)
  %i2r, %i2r_flags = hardfloat.in_to_rec_fn<24, 8, 32>(%false, %val_i32, %rm, %false) : (i1, i32, i3, i1) -> (i33, i5)

  // 6. Recoded to Integer: (in, rm, signedOut) -> (out, flags)
  // Note: exceptionFlags is i3 for this op
  %r2i, %r2i_flags = hardfloat.rec_fn_to_in<24, 8, 32>(%rec, %rm, %true) : (i33, i3, i1) -> (i32, i3)

  func.return
}

// CHECK:       func.func @test_hardfloat_roundtrip(%{{.*}}: i33, %{{.*}}: i33, %{{.*}}: i32) {
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    %rm = arith.constant 0 : i3
// CHECK-NEXT:    %mul, %m_flags = hardfloat.mul_rec_fn<24, 8>(%a, %b, %rm, %false) : (i33, i33, i3, i1) -> (i33, i5)
// CHECK-NEXT:    %add, %a_flags = hardfloat.add_rec_fn<24, 8>(%false, %a, %b, %rm, %false) : (i1, i33, i33, i3, i1) -> (i33, i5)
// CHECK-NEXT:    %rec = hardfloat.fn_to_rec_fn<24, 8>(%val_i32) : (i32) -> i33
// CHECK-NEXT:    %f_out = hardfloat.rec_fn_to_fn<24, 8>(%rec) : (i33) -> i32
// CHECK-NEXT:    %i2r, %i2r_flags = hardfloat.in_to_rec_fn<24, 8, 32>(%false, %val_i32, %rm, %false) : (i1, i32, i3, i1) -> (i33, i5)
// CHECK-NEXT:    %r2i, %r2i_flags = hardfloat.rec_fn_to_in<24, 8, 32>(%rec, %rm, %true) : (i33, i3, i1) -> (i32, i3)
// CHECK-NEXT:    return
// CHECK-NEXT:  }
