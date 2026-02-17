// RUN: XDSL_ROUNDTRIP

func.func @test_mul(%a : i33, %b : i33, %c : i32) -> (i33, i33, i32) {
  %mul = hardfloat.mul<24, 8>(%a, %b) : (i33, i33) -> i33
  %add = hardfloat.add<24, 8>(%a, %b) : (i33, i33) -> i33
  %recoded = hardfloat.recode<24, 8>(%c) : (i32) -> i33
  %unrecoded = hardfloat.unrecode<24, 8>(%recoded) : (i33) -> i32
  func.return %mul, %add, %unrecoded : i33, i33, i32
}

// CHECK: func.func @test_mul(%a : i33, %b : i33, %c : i32) -> (i33, i33, i32) {
// CHECK-NEXT:   %mul = hardfloat.mul<24, 8>(%a, %b) : (i33, i33) -> i33
// CHECK-NEXT:   %add = hardfloat.add<24, 8>(%a, %b) : (i33, i33) -> i33
// CHECK-NEXT:   %recoded = hardfloat.recode<24, 8>(%c) : (i32) -> i33
// CHECK-NEXT:   %unrecoded = hardfloat.unrecode<24, 8>(%recoded) : (i33) -> i32
// CHECK-NEXT:   func.return %mul, %add, %unrecoded : i33, i33, i32
// CHECK-NEXT: }
