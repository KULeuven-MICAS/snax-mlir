// RUN: ./compiler/snax-opt -p convert-linalg-to-stream %s | filecheck %s

%arg0, %arg1, %arg2 = "test.op"() : () -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>)
%c0_i32 = arith.constant 0 : i32
// CHECK: builtin.module {
// CHECK-NEXT:  %arg0, %arg1, %arg2 = "test.op"() : () -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>)
// CHECK-NEXT:  %c0_i32 = arith.constant 0 : i32

%0 = tensor.empty() : tensor<16x16xi32>
// CHECK-NEXT:  %0 = tensor.empty() : tensor<16x16xi32>

%1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], library_call = "snax_gemmx_stream"} ins(%arg0, %arg1, %c0_i32, %c0_i32 : tensor<16x16xi8>, tensor<16x16xi8>, i32, i32) outs(%0 : tensor<16x16xi32>) {
^0(%in : i8, %in_1 : i8, %in_2 : i32, %in_3 : i32, %out : i32):
  %2 = kernel.qmac %in, %in_1 zp_lhs : %in_2 zp_rhs : %in_3 : i8, i8, i32, i32 -> i32
  linalg.yield %2 : i32
} -> tensor<16x16xi32>
// CHECK-NEXT: %1 = "stream.streaming_region"(%arg0, %arg1, %0) <{"patterns" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>], "accelerator" = "snax_gemmx", "operandSegmentSizes" = array<i32: 2, 1>}> ({
// CHECK-NEXT: ^0(%2 : !stream.stream<i8>, %3 : !stream.stream<i8>, %4 : !stream.stream<i32>):
// CHECK-NEXT:   %5 = "stream.generic"(%2, %3, %c0_i32, %c0_i32) <{"library_call" = "snax_gemmx", "operandSegmentSizes" = array<i32: 4>}> ({
// CHECK-NEXT:   ^1(%in : i8, %in_1 : i8, %in_2 : i32, %in_3 : i32, %out : i32):
// CHECK-NEXT:     %6 = kernel.qmac %in, %in_1 zp_lhs : %in_2 zp_rhs : %in_3 : i8, i8, i32, i32 -> i32
// CHECK-NEXT:     stream.yield %6 : i32
// CHECK-NEXT:   }) : (!stream.stream<i8>, !stream.stream<i8>, i32, i32) -> !stream.stream<i32>
// CHECK-NEXT:   stream.yield %5 : !stream.stream<i32>
// CHECK-NEXT: }) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>

%3 = tensor.empty() : tensor<16x16xi32>
//CHECK-NEXT:   %7 = tensor.empty() : tensor<16x16xi32>

%4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"], library_call = "snax_gemmx_stream"} ins(%1, %arg2 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%3 : tensor<16x16xi32>) {
^1(%in_4 : i32, %in_5 : i32, %out_1 : i32):
  %5 = kernel.add %in_4, %in_5 : i32, i32 -> i32
  linalg.yield %5 : i32
} -> tensor<16x16xi32>

//CHECK-NEXT: %8 = "stream.streaming_region"(%1, %arg2, %7) <{"patterns" = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], "accelerator" = "snax_gemmx", "operandSegmentSizes" = array<i32: 2, 1>}> ({
//CHECK-NEXT: ^2(%9 : !stream.stream<i32>, %10 : !stream.stream<i32>, %11 : !stream.stream<i32>):
//CHECK-NEXT:   %12 = "stream.generic"(%9, %10) <{"library_call" = "snax_gemmx", "operandSegmentSizes" = array<i32: 2>}> ({
//CHECK-NEXT:   ^3(%in_4 : i32, %in_5 : i32, %out_1 : i32):
//CHECK-NEXT:     %13 = kernel.add %in_4, %in_5 : i32, i32 -> i32
//CHECK-NEXT:     stream.yield %13 : i32
//CHECK-NEXT:   }) : (!stream.stream<i32>, !stream.stream<i32>) -> !stream.stream<i32>
//CHECK-NEXT:   stream.yield %12 : !stream.stream<i32>
//CHECK-NEXT: }) : (tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>


