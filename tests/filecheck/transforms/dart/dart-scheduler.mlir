// RUN: ./compiler/snax-opt --split-input-file %s -p insert-accfg-op{accelerator=snax_gemmx},dart-scheduler | filecheck %s

func.func @streamer_matmul(%arg0 : memref<16x16xi8>, %arg1 : memref<16x16xi8, strided<[1, 16]>>, %arg2 : memref<16x16xi32>) {
  %0 = arith.constant 0 : i32
  "dart.operation"(%arg0, %arg1, %arg2) <{patterns = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], accelerator = "snax_gemmx", operandSegmentSizes = array<i32: 2, 1>}> ({
  ^0(%1 : !dart.stream<i8>, %2 : !dart.stream<i8>, %3 : !dart.stream<i32>):
    %4 = "dart.generic"(%1, %2, %0, %0) <{library_call = "snax_gemmx"}> ({
    ^1(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
      %5 = kernel.qmac %arg3, %arg4 zp_lhs : %arg5 zp_rhs : %arg6 : i8, i8, i32, i32 -> i32
      dart.yield %5 : i32
    }) : (!dart.stream<i8>, !dart.stream<i8>, i32, i32) -> !dart.stream<i32>
    dart.yield %4 : !dart.stream<i32>
  }) : (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32>) -> ()
  func.return
}

// CHECK:     "dart.schedule"(%arg0, %arg1, %arg2) <{patterns = [affine_map<(d0, d1, d2, d3, d4, d5) -> (((d1 * 8) + d3), ((d2 * 8) + d5))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (((d2 * 8) + d5), ((d0 * 8) + d4))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (((d1 * 8) + d3), ((d0 * 8) + d4))>], accelerator = "snax_gemmx", tiles = [[]], bounds = [2 : index, 2 : index, 2 : index, 8 : index, 8 : index, 8 : index], operandSegmentSizes = array<i32: 2, 1>}> (
// CHECK-NEXT:     ^0(%1 : !dart.stream<i8>, %2 : !dart.stream<i8>, %3 : !dart.stream<i32>):
// CHECK-NEXT:       %4 = "dart.generic"(%1, %2, %0, %0) <{library_call = "snax_gemmx"}> ({
// CHECK-NEXT:       ^1(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
// CHECK-NEXT:         %5 = kernel.qmac %arg3, %arg4 zp_lhs : %arg5 zp_rhs : %arg6 : i8, i8, i32, i32 -> i32
// CHECK-NEXT:         dart.yield %5 : i32
// CHECK-NEXT:       }) : (!dart.stream<i8>, !dart.stream<i8>, i32, i32) -> !dart.stream<i32>
// CHECK-NEXT:       dart.yield %4 : !dart.stream<i32>
// CHECK-NEXT:     }) : (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32>) -> ()
