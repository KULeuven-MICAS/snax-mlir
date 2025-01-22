// RUN: ./compiler/snax-opt --split-input-file %s -p set-memory-layout --print-op-generic | filecheck %s
// RUN: ./compiler/snax-opt --split-input-file %s -p set-memory-layout{gemm_layout=banked} --print-op-generic | filecheck %s --check-prefix=BANKED

func.func @gemm(%arg0 : memref<16x16xi8, "L1">, %arg1 : memref<16x16xi8, "L1">, %arg2 : memref<16x16xi32, "L1">) -> () {
  %0 = arith.constant 0 : i32
  %1 = arith.constant 0 : i32
  "dart.operation"(%arg0, %arg1, %arg2) <{"accelerator" = "snax_gemmx", "operandSegmentSizes" = array<i32: 2, 1>, "patterns" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
  ^0(%arg3 : !dart.stream<i8>, %arg4 : !dart.stream<i8>, %arg5 : !dart.stream<i32>):
    %5 = "dart.generic"(%arg3, %arg4, %0, %1) <{"library_call" = "snax_gemmx"}> ({
    ^1(%arg6 : i8, %arg7 : i8, %arg8 : i32, %arg9 : i32, %arg10 : i32):
      %6 = kernel.qmac %arg6, %arg7 zp_lhs : %arg8 zp_rhs : %arg9 : i8, i8, i32, i32 -> i32
      dart.yield %6 : i32
    }) : (!dart.stream<i8>, !dart.stream<i8>, i32, i32) -> !dart.stream<i32>
    dart.yield %5 : !dart.stream<i32>
  }) : (memref<16x16xi8, "L1">, memref<16x16xi8, "L1">, memref<16x16xi32, "L1">) -> ()
  func.return
}

// CHECK:      %2 = "snax.layout_cast"(%arg0) : (memref<16x16xi8, "L1">) -> memref<16x16xi8, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>, "L1">
// CHECK-NEXT: %3 = "snax.layout_cast"(%arg1) : (memref<16x16xi8, "L1">) -> memref<16x16xi8, #tsl.tsl<[2, 8] -> (64, 1), [2, 8] -> (128, 8)>, "L1">
// CHECK-NEXT: %4 = "snax.layout_cast"(%arg2) : (memref<16x16xi32, "L1">) -> memref<16x16xi32, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>, "L1">

// BANKED:       %2 = "snax.layout_cast"(%arg0) : (memref<16x16xi8, "L1">) -> memref<16x16xi8, #tsl.tsl<[2, 8] -> (512, 8), [2, 8] -> (256, 1)>, "L1">
// BANKED-NEXT:  %3 = "snax.layout_cast"(%arg1) : (memref<16x16xi8, "L1">) -> memref<16x16xi8, #tsl.tsl<[2, 8] -> (256, 1), [2, 8] -> (512, 8)>, "L1">
// BANKED-NEXT:  %4 = "snax.layout_cast"(%arg2) : (memref<16x16xi32, "L1">) -> memref<16x16xi32, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>, "L1">
