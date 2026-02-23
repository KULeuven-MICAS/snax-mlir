// REQUIRES: has_easyfloat_installed
// RUN: snax-opt -p convert-hardfloat-to-hw{'easyfloat_path="%p/../../../../../kuleuven-easyfloat"'} %s | filecheck %s
// RUN: snax-opt -p convert-hardfloat-to-hw{external_modules=true} %s | filecheck %s --check-prefix=EXTERN


func.func @test_hardfloat(%a : f32, %b : f32) -> f32 {
  %false = arith.constant false
  %rm = arith.constant 0 : i3
  %a_i32 = builtin.unrealized_conversion_cast %a : f32 to i32
  %b_i32 = builtin.unrealized_conversion_cast %b : f32 to i32
  %a_rec = hardfloat.fn_to_rec_fn<24, 8>(%a_i32) : (i32) -> i33
  %b_rec = hardfloat.fn_to_rec_fn<24, 8>(%b_i32) : (i32) -> i33
  %add_res, %add_flags = hardfloat.add_rec_fn<24, 8>(%false, %a_rec, %b_rec, %rm, %false) : (i1, i33, i33, i3, i1) -> (i33, i5)
  %mul_res, %mul_flags = hardfloat.mul_rec_fn<24, 8>(%add_res, %b_rec, %rm, %false) : (i33, i33, i3, i1) -> (i33, i5)
  %mul_val_i32 = hardfloat.rec_fn_to_fn<24, 8>(%mul_res) : (i33) -> i32
  %final_f32 = builtin.unrealized_conversion_cast %mul_val_i32 : i32 to f32
  func.return %final_f32 : f32
}

// CHECK: func.func @test_hardfloat(%a : f32, %b : f32) -> f32 {
// CHECK-NEXT:   %false = arith.constant false
// CHECK-NEXT:   %rm = arith.constant 0 : i3
// CHECK-NEXT:   %a_i32 = builtin.unrealized_conversion_cast %a : f32 to i32
// CHECK-NEXT:   %b_i32 = builtin.unrealized_conversion_cast %b : f32 to i32
// CHECK-NEXT:   %a_rec = hw.instance "recFNFromFN_s24_e8_0" @recFNFromFN_s24_e8(io_in: %a_i32: i32) -> (io_out: i33)
// CHECK-NEXT:   %b_rec = hw.instance "recFNFromFN_s24_e8_1" @recFNFromFN_s24_e8(io_in: %b_i32: i32) -> (io_out: i33)
// CHECK-NEXT:   %add_res, %add_flags = hw.instance "AddRecFN_s24_e8_0" @AddRecFN_s24_e8(io_subOp: %false: i1, io_a: %a_rec: i33, io_b: %b_rec: i33, io_roundingMode: %rm: i3, io_detectTininess: %false: i1) -> (io_out: i33, io_exceptionFlags: i5)
// CHECK-NEXT:   %mul_res, %mul_flags = hw.instance "MulRecFN_s24_e8_0" @MulRecFN_s24_e8(io_a: %add_res: i33, io_b: %b_rec: i33, io_roundingMode: %rm: i3, io_detectTininess: %false: i1) -> (io_out: i33, io_exceptionFlags: i5)
// CHECK-NEXT:   %mul_val_i32 = hw.instance "fNFromRecFN_s24_e8_0" @fNFromRecFN_s24_e8(io_in: %mul_res: i33) -> (io_out: i32)
// CHECK-NEXT:   %final_f32 = builtin.unrealized_conversion_cast %mul_val_i32 : i32 to f32
// CHECK-NEXT:   func.return %final_f32 : f32
// CHECK-NEXT: }
// CHECK: hw.module private @recFNFromFN_s24_e8(in %io_in: i32, out io_out: i33) {
// CHECK: hw.module private @AddRawFN(in %io_subOp: i1, in %io_a_isNaN: i1, in %io_a_isInf: i1, in %io_a_isZero: i1, in %io_a_sign: i1, in %io_a_sExp: i10, in %io_a_sig: i25, in %io_b_isNaN: i1, in %io_b_isInf: i1, in %io_b_isZero: i1, in %io_b_sign: i1, in %io_b_sExp: i10, in %io_b_sig: i25, in %io_roundingMode: i3, out io_invalidExc: i1, out io_rawOut_isNaN: i1, out io_rawOut_isInf: i1, out io_rawOut_isZero: i1, out io_rawOut_sign: i1, out io_rawOut_sExp: i10, out io_rawOut_sig: i27) {
// CHECK: hw.module private @RoundAnyRawFNToRecFN_ie8_is26_oe8_os24(in %io_invalidExc: i1, in %io_in_isNaN: i1, in %io_in_isInf: i1, in %io_in_isZero: i1, in %io_in_sign: i1, in %io_in_sExp: i10, in %io_in_sig: i27, in %io_roundingMode: i3, in %io_detectTininess: i1, out io_out: i33, out io_exceptionFlags: i5) {
// CHECK: hw.module private @RoundRawFNToRecFN_e8_s24(in %io_invalidExc: i1, in %io_in_isNaN: i1, in %io_in_isInf: i1, in %io_in_isZero: i1, in %io_in_sign: i1, in %io_in_sExp: i10, in %io_in_sig: i27, in %io_roundingMode: i3, in %io_detectTininess: i1, out io_out: i33, out io_exceptionFlags: i5) {
// CHECK: hw.module private @AddRecFN_s24_e8(in %io_subOp: i1, in %io_a: i33, in %io_b: i33, in %io_roundingMode: i3, in %io_detectTininess: i1, out io_out: i33, out io_exceptionFlags: i5) {
// CHECK: hw.module private @MulFullRawFN(in %io_a_isNaN: i1, in %io_a_isInf: i1, in %io_a_isZero: i1, in %io_a_sign: i1, in %io_a_sExp: i10, in %io_a_sig: i25, in %io_b_isNaN: i1, in %io_b_isInf: i1, in %io_b_isZero: i1, in %io_b_sign: i1, in %io_b_sExp: i10, in %io_b_sig: i25, out io_invalidExc: i1, out io_rawOut_isNaN: i1, out io_rawOut_isInf: i1, out io_rawOut_isZero: i1, out io_rawOut_sign: i1, out io_rawOut_sExp: i10, out io_rawOut_sig: i48) {
// CHECK: hw.module private @MulRawFN(in %io_a_isNaN: i1, in %io_a_isInf: i1, in %io_a_isZero: i1, in %io_a_sign: i1, in %io_a_sExp: i10, in %io_a_sig: i25, in %io_b_isNaN: i1, in %io_b_isInf: i1, in %io_b_isZero: i1, in %io_b_sign: i1, in %io_b_sExp: i10, in %io_b_sig: i25, out io_invalidExc: i1, out io_rawOut_isNaN: i1, out io_rawOut_isInf: i1, out io_rawOut_isZero: i1, out io_rawOut_sign: i1, out io_rawOut_sExp: i10, out io_rawOut_sig: i27) {
// CHECK: hw.module private @MulRecFN_s24_e8(in %io_a: i33, in %io_b: i33, in %io_roundingMode: i3, in %io_detectTininess: i1, out io_out: i33, out io_exceptionFlags: i5) {
// CHECK: hw.module private @fNFromRecFN_s24_e8(in %io_in: i33, out io_out: i32) {

// EXTERN: hw.module.extern @AddRecFN_s24_e8(in %port0 io_subOp: i1, in %port1 io_a: i33, in %port2 io_b: i33, in %port3 io_roundingMode: i3, in %port4 io_detectTininess: i1, out io_out: i33, out io_exceptionFlags: i5)
// EXTERN-NEXT: hw.module.extern @MulRecFN_s24_e8(in %port0 io_a: i33, in %port1 io_b: i33, in %port2 io_roundingMode: i3, in %port3 io_detectTininess: i1, out io_out: i33, out io_exceptionFlags: i5)
// EXTERN-NEXT: hw.module.extern @fNFromRecFN_s24_e8(in %port0 io_in: i33, out io_out: i32)
// EXTERN-NEXT: hw.module.extern @recFNFromFN_s24_e8(in %port0 io_in: i32, out io_out: i33)
