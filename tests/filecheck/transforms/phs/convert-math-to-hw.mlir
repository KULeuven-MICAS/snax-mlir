// RUN: snax-opt -p phs-convert-float-to-hardfloat %s | filecheck %s


func.func @test_addf (%1: f32, %2: f32) -> f32 {
  %add = arith.addf %1, %2 : f32
  return %add : f32
}

func.func @test_mulf (%1 : f32, %2 : f32) -> f32 {
  %mul = arith.mulf %1, %2 : f32
  return %mul : f32
}

func.func @test_fused (%0 : f32, %1: f32) -> f32 {
  %mul = arith.mulf %0, %0 : f32
  %add = arith.addf %mul, %1 : f32
  return %add : f32
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @test_addf(%0 : f32, %1 : f32) -> f32 {
//
// We should see a set of uncrealized conversion casts, followed by recodes for each operand
// CHECK-NEXT:     %add = builtin.unrealized_conversion_cast %0 : f32 to i32
// CHECK-NEXT:     %add_1 = hw.instance "recoder_lhs" @RecFNFromFN(io_in: %add: i32) -> (io_out: i33)
// CHECK-NEXT:     %add_2 = builtin.unrealized_conversion_cast %1 : f32 to i32
// CHECK-NEXT:     %add_3 = hw.instance "recoder_rhs" @RecFNFromFN(io_in: %add_2: i32) -> (io_out: i33)
//
// Now we should see a call to AddRecFN with the recoded i33s:
// CHECK-NEXT:     %add_4 = hw.instance "arith.addf" @AddRecFN(io_a: %add_1: i33, io_b: %add_3: i33) -> (io_out: i33)
//
// And finally the result should be unrecoded back to i32:
// CHECK-NEXT:     %add_5 = hw.instance "unrecoderr" @fNFromRecFN(io_in: %add_4: i33) -> (io_out: i32)
// CHECK-NEXT:     %add_6 = builtin.unrealized_conversion_cast %add_5 : i32 to f32
//
// And returned
// CHECK-NEXT:     func.return %add_6 : f32
// CHECK-NEXT:   }

// same here:
// CHECK-NEXT:  func.func @test_mulf(%0 : f32, %1 : f32) -> f32 {
// CHECK-NEXT:    %mul = builtin.unrealized_conversion_cast %0 : f32 to i32
// CHECK-NEXT:    %mul_1 = hw.instance "recoder_lhs" @RecFNFromFN(io_in: %mul: i32) -> (io_out: i33)
// CHECK-NEXT:    %mul_2 = builtin.unrealized_conversion_cast %1 : f32 to i32
// CHECK-NEXT:    %mul_3 = hw.instance "recoder_rhs" @RecFNFromFN(io_in: %mul_2: i32) -> (io_out: i33)
// CHECK-NEXT:    %mul_4 = hw.instance "arith.mulf" @MulRecFN(io_a: %mul_1: i33, io_b: %mul_3: i33) -> (io_out: i33)
// CHECK-NEXT:    %mul_5 = hw.instance "unrecoderr" @fNFromRecFN(io_in: %mul_4: i33) -> (io_out: i32)
// CHECK-NEXT:    %mul_6 = builtin.unrealized_conversion_cast %mul_5 : i32 to f32
// CHECK-NEXT:    func.return %mul_6 : f32
// CHECK-NEXT:  }

// check that unrecode->recode pairs cancel out properly
// also check that op(%a, %a) only results in %a being recoded once
// CHECK-NEXT:  func.func @test_fused(%0 : f32, %1 : f32) -> f32 {
//
// here we see only one recode, because the mulf takes %0 twice
// CHECK-NEXT:    %mul = builtin.unrealized_conversion_cast %0 : f32 to i32
// CHECK-NEXT:    %mul_1 = hw.instance "recoder_lhs" @RecFNFromFN(io_in: %mul: i32) -> (io_out: i33)
//
// mulf with %mul_1 arg twice:
// CHECK-NEXT:    %mul_2 = hw.instance "arith.mulf" @MulRecFN(io_a: %mul_1: i33, io_b: %mul_1: i33) -> (io_out: i33)
// missing output cast:
//
// instead this is the cast for %1 now, which is the second arg to the addf:
// CHECK-NEXT:    %add = builtin.unrealized_conversion_cast %1 : f32 to i32
// CHECK-NEXT:    %add_1 = hw.instance "recoder_rhs" @RecFNFromFN(io_in: %add: i32) -> (io_out: i33)
//
// addf gets the result of the mulf (%mul_2) directly: no conversion at all:
// CHECK-NEXT:    %add_2 = hw.instance "arith.addf" @AddRecFN(io_a: %mul_2: i33, io_b: %add_1: i33) -> (io_out: i33)
//
// now we finally unrecode the result:
// CHECK-NEXT:    %add_3 = hw.instance "unrecoderr" @fNFromRecFN(io_in: %add_2: i33) -> (io_out: i32)
// CHECK-NEXT:    %add_4 = builtin.unrealized_conversion_cast %add_3 : i32 to f32
// CHECK-NEXT:    func.return %add_4 : f32
// CHECK-NEXT:  }

// finaly, all the external modules, sorted alphabetically
// CHECK-NEXT:   hw.module.extern @AddRecFN(in %port0 io_a: i33, in %port1 io_b: i33, out io_out: i33)
// CHECK-NEXT:   hw.module.extern @MulRecFN(in %port0 io_a: i33, in %port1 io_b: i33, out io_out: i33)
// CHECK-NEXT:   hw.module.extern @RecFNFromFN(in %port0 io_in: i32, out io_out: i33)
// CHECK-NEXT:   hw.module.extern @fNFromRecFN(in %port0 io_in: i33, out io_out: i32)
// CHECK-NEXT: }
