// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_SINGLETRIP

%s0, %s1, %s2 = "test.op"() : () -> (!stream.stream<i8>, !stream.stream<i32>, !stream.stream<f32>)
%t0, %t1 = "test.op"() : () -> (tensor<16x16xi8>, tensor<16x16xi32>)

%0 = "stream.streaming_region"(%t0, %t0, %t1) <{"patterns" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>], "accelerator" = "snax_gemmx_stream", "operandSegmentSizes" = array<i32: 2, 1>}> ({
^0(%1 : !stream.stream<i8>, %2 : !stream.stream<i8>, %3 : !stream.stream<i32>):
  %4 = "stream.generic"(%1, %2) ({
  ^1(%in : i8, %in_1 : i8):
    %5 = "test.op"(%in, %in_1) : (i8, i8) -> i32
    stream.yield %5 : i32
  }) : (!stream.stream<i8>, !stream.stream<i8>) -> !stream.stream<i32>
  stream.yield %4 : !stream.stream<i32>
}) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>


// CHECK: builtin.module {
// CHECK-NEXT:   %s0, %s1, %s2 = "test.op"() : () -> (!stream.stream<i8>, !stream.stream<i32>, !stream.stream<f32>)
// CHECK-NEXT:   %t0, %t1 = "test.op"() : () -> (tensor<16x16xi8>, tensor<16x16xi32>)
// CHECK-NEXT:   %0 = "stream.streaming_region"(%t0, %t0, %t1) <{"patterns" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>], "accelerator" = "snax_gemmx_stream", "operandSegmentSizes" = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^0(%1 : !stream.stream<i8>, %2 : !stream.stream<i8>, %3 : !stream.stream<i32>):
// CHECK-NEXT:     %4 = "stream.generic"(%1, %2) ({
// CHECK-NEXT:     ^1(%in : i8, %in_1 : i8):
// CHECK-NEXT:       %5 = "test.op"(%in, %in_1) : (i8, i8) -> i32
// CHECK-NEXT:       stream.yield %5 : i32
// CHECK-NEXT:     }) : (!stream.stream<i8>, !stream.stream<i8>) -> !stream.stream<i32>
// CHECK-NEXT:     stream.yield %4 : !stream.stream<i32>
// CHECK-NEXT:   }) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>
// CHECK-NEXT: }
