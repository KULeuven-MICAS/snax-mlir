// RUN: snax-opt %s --verify-diagnostics --split-input-file | filecheck %s

// 1. Test RecodedInputs: Input bitwidth != sig_width + exp_width + 1
func.func @test_recoded_inputs_width(%a : i32, %b : i33) {
  // CHECK: Expect input type to be of sig_width (24) + exp_width (8) + 1 = 32
  %mul = hardfloat.mul_rec_fn<24, 8>(%a, %b) : (i32, i33) -> i33
  func.return
}

// -----

// 2. Test RecodedInputs: Input is not an IntegerType (if applicable in IR parsing)
// Note: In xDSL, if the operand_def is IntegerType, the parser might catch it first,
// but the verifier trait ensures it.
func.func @test_recoded_inputs_type(%a : f32, %b : i33) {
  // CHECK: operand 'lhs' at position 0 does not verify:
  // CHECK-NEXT: f32 should be of base attribute integer_type
  %mul = hardfloat.mul_rec_fn<24, 8>(%a, %b) : (f32, i33) -> i33
  func.return
}

// -----

// 3. Test RecodedOutputs: Output bitwidth != sig_width + exp_width + 1
func.func @test_recoded_outputs_width(%a : i33, %b : i33) {
  // expected-error@+1 {{Expect output type to be of sig_width (24) + exp_width (8) + 1 = 32}}
  %mul = hardfloat.mul_rec_fn<24, 8>(%a, %b) : (i33, i33) -> i32
  func.return
}

// -----

// 4. Test IntegerOutputs: Output bitwidth != int_width
func.func @test_int_outputs_width(%a : i33) {
  // expected-error@+1 {{Expect output type (i16) to have bitwidth given by int_width property (32)}}
  %to_int = hardfloat.rec_fn_to_in<24, 8, 32>(%a) <{signedness=#builtin.signedness<unsigned>}>: (i33) -> i16
  func.return
}

// -----

// 5. Test IntegerInputs: Input bitwidth != int_width
func.func @test_int_inputs_width(%a : i16) {
  // expected-error@+1 {{Expect input type (i16) to have bitwidth given by int_width property (32)}}
  %int_to_rec = hardfloat.in_to_rec_fn<24, 8, 32>(%a) <{signedness=#builtin.signedness<unsigned>}>: (i16) -> i33
  func.return
}

// -----

// 6. Test IntegerConversion: Signedness cannot be Signless
func.func @test_signless_conversion(%a : i33) {
  // expected-error@+1 {{Property signedness can not be Signless}}
  %to_int = hardfloat.rec_fn_to_in<24, 8, 32>(%a) <{signedness=#builtin.signedness<signless>}>: (i33) -> i32
  func.return
}

// -----

// 7. Test IntegerConversion: Missing signedness property
// Note: This assumes the parser allows a missing property; if prop_def is mandatory,
// the parser will fail, but if forced, the trait verifier catches it.
"func.func"() ({
  %0 = "test.op"() : () -> i33
  // expected-error@+1 {{IntegerConversion optrait expects signedness attr}}
  %1 = "hardfloat.rec_fn_to_in"(%0) {"exp_width" = 8 : i64, "int_width" = 32 : i64, "sig_width" = 24 : i64} : (i33) -> i32
  "func.return"() : () -> ()
}) : () -> ()
