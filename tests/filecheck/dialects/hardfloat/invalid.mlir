// RUN: snax-opt %s --verify-diagnostics --split-input-file | filecheck %s

// 1. Test verify_recoded: Input/Output bitwidth != sig_width + exp_width + 1
func.func @test_verify_recoded(%a : i32, %b : i33, %rm : i3, %tininess : i1) {
  // CHECK: Expect type (i32) to be equal to sig_width (24) + exp_width (8) + 1
  %mul = hardfloat.mul_rec_fn<24, 8>(%a, %b, %rm, %tininess) : (i32, i33, i3, i1) -> i33
  func.return
}

// -----

// 2. Test verify_float: Input bitwidth != sig_width + exp_width
func.func @test_verify_float(%a : i31) {
  // CHECK: Expect type (i31) to be equal to sig_width (24) + exp_width (8)
  %recoded = hardfloat.fn_to_rec_fn<24, 8>(%a) : (i31) -> i33
  func.return
}

// -----

// 3. Test verify_int: Missing int_width property (Parsing should fail or verify catch)
func.func @test_missing_int_width(%a : i33, %signed : i1) {
  // If the parser allowed skipping the third param, the verifier catches the missing int_width
  // CHECK: Expect op to have int_width property
  %to_int = hardfloat.rec_fn_to_in<24, 8>(%a, %signed) : (i33, i1) -> i32
  func.return
}

// -----

// 4. Test verify_int: bitwidth != int_width.data
func.func @test_int_width_mismatch(%a : i33, %signed : i1) {
  // CHECK: Expect output type (i16) to have bitwidth given by int_width property (32)
  %to_int = hardfloat.rec_fn_to_in<24, 8, 32>(%a, %signed) : (i33, i1) -> i16
  func.return
}

// -----

// 5. Test AddRecFnOp: Specific check on one of the new operands (subOp)
func.func @test_add_subop_type(%a : i33, %b : i33, %rm : i3, %tininess : i1, %bad_sub : i32) {
  // IRDL verifier will catch the i32 where i1 is expected for subOp
  // CHECK: Operation does not verify: operand 'subOp' at position 0 does not verify:
  // CHECK-NEXT: Expected attribute i1 but got i32
  %add = "hardfloat.add_rec_fn"(%bad_sub, %a, %b, %rm, %tininess) {"exp_width" = 8 : i64, "sig_width" = 24 : i64} : (i32, i33, i33, i3, i1) -> i33
  func.return
}

// -----

// 6. Test verify_recoded on Result: Output bitwidth mismatch
func.func @test_recoded_result_mismatch(%a : i32, %signed : i1) {
  // CHECK: Expect type (i16) to be equal to sig_width (24) + exp_width (8) + 1
  %int_to_rec = hardfloat.in_to_rec_fn<24, 8, 32>(%signed, %a) : (i1, i32) -> i16
  func.return
}
