// RUN: ./compiler/snax-opt -p fuse-streaming-regions %s | filecheck %s

func.func @streamer_matmul(%arg0 : tensor<16x16xi8>, %arg1 : tensor<16x16xi8>, %arg2 : tensor<16x16xi32>) -> tensor<16x16xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<16x16xi32>

  // first streaming region (matmul):
  %1 = "stream.streaming_region"(%arg0, %arg1, %0) <{"patterns" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>], "accelerator" = "snax_gemmx", "operandSegmentSizes" = array<i32: 2, 1>}> ({
  ^0(%2 : !stream.stream<i8>, %3 : !stream.stream<i8>, %4 : !stream.stream<i32>):
    %5 = "stream.generic"(%2, %3, %c0_i32, %c0_i32) <{"library_call" = "snax_gemmx"}> ({
    ^1(%in : i8, %in_1 : i8, %in_2 : i32, %in_3 : i32, %out : i32):
      %6 = kernel.qmac %in, %in_1 zp_lhs : %in_2 zp_rhs : %in_3 : i8, i8, i32, i32 -> i32
      stream.yield %6 : i32
    }) : (!stream.stream<i8>, !stream.stream<i8>, i32, i32) -> !stream.stream<i32>
    stream.yield %5 : !stream.stream<i32>
  }) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>

  %7 = tensor.empty() : tensor<16x16xi32>

  // second streaming region (elementwise add):
  %8 = "stream.streaming_region"(%1, %arg2, %7) <{"patterns" = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], "accelerator" = "snax_gemmx", "operandSegmentSizes" = array<i32: 2, 1>}> ({
  ^2(%9 : !stream.stream<i32>, %10 : !stream.stream<i32>, %11 : !stream.stream<i32>):
    %12 = "stream.generic"(%9, %10) <{"library_call" = "snax_gemmx"}> ({
    ^3(%in_4 : i32, %in_5 : i32, %out_1 : i32):
      %13 = kernel.add %in_4, %in_5 : i32, i32 -> i32
      stream.yield %13 : i32
    }) : (!stream.stream<i32>, !stream.stream<i32>) -> !stream.stream<i32>
    stream.yield %12 : !stream.stream<i32>
  }) : (tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>

  %13 = tensor.empty() : tensor<16x16xi32>

  // third streaming region (elementwise add):
  %14 = "stream.streaming_region"(%8, %arg2, %13) <{"patterns" = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], "accelerator" = "snax_gemmx", "operandSegmentSizes" = array<i32: 2, 1>}> ({
  ^2(%15 : !stream.stream<i32>, %16 : !stream.stream<i32>, %17 : !stream.stream<i32>):
    %18 = "stream.generic"(%15, %16) <{"library_call" = "snax_gemmx"}> ({
    ^3(%in_6 : i32, %in_7 : i32, %out_2 : i32):
      %19 = kernel.add %in_6, %in_7 : i32, i32 -> i32
      stream.yield %19 : i32
    }) : (!stream.stream<i32>, !stream.stream<i32>) -> !stream.stream<i32>
    stream.yield %18 : !stream.stream<i32>
  }) : (tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>

  func.return %14 : tensor<16x16xi32>
}


// CHECK: builtin.module {
// CHECK-NEXT:   func.func @streamer_matmul(%arg0 : tensor<16x16xi8>, %arg1 : tensor<16x16xi8>, %arg2 : tensor<16x16xi32>) -> tensor<16x16xi32> {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = tensor.empty() : tensor<16x16xi32>
// CHECK-NEXT:     %1 = tensor.empty() : tensor<16x16xi32>
// CHECK-NEXT:     %2 = tensor.empty() : tensor<16x16xi32>
// CHECK-NEXT:     %3 = "stream.streaming_region"(%arg0, %arg1, %arg2, %arg2, %2) <{"patterns" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], "operandSegmentSizes" = array<i32: 4, 1>}> ({
// CHECK-NEXT:     ^0(%4 : !stream.stream<i8>, %5 : !stream.stream<i8>, %6 : !stream.stream<i32>, %7 : !stream.stream<i32>, %8 : !stream.stream<i32>):
// CHECK-NEXT:       %9 = "stream.generic"(%4, %5, %c0_i32, %c0_i32) <{"library_call" = "snax_gemmx"}> ({
// CHECK-NEXT:       ^1(%in : i8, %in_1 : i8, %in_2 : i32, %in_3 : i32, %out : i32):
// CHECK-NEXT:         %10 = kernel.qmac %in, %in_1 zp_lhs : %in_2 zp_rhs : %in_3 : i8, i8, i32, i32 -> i32
// CHECK-NEXT:         stream.yield %10 : i32
// CHECK-NEXT:       }) : (!stream.stream<i8>, !stream.stream<i8>, i32, i32) -> !stream.stream<i32>
// CHECK-NEXT:       %11 = "stream.generic"(%9, %6) <{"library_call" = "snax_gemmx"}> ({
// CHECK-NEXT:       ^2(%in_4 : i32, %in_5 : i32, %out_1 : i32):
// CHECK-NEXT:         %12 = kernel.add %in_4, %in_5 : i32, i32 -> i32
// CHECK-NEXT:         stream.yield %12 : i32
// CHECK-NEXT:       }) : (!stream.stream<i32>, !stream.stream<i32>) -> !stream.stream<i32>
// CHECK-NEXT:       %13 = "stream.generic"(%11, %7) <{"library_call" = "snax_gemmx"}> ({
// CHECK-NEXT:       ^3(%in_6 : i32, %in_7 : i32, %out_2 : i32):
// CHECK-NEXT:         %14 = kernel.add %in_6, %in_7 : i32, i32 -> i32
// CHECK-NEXT:         stream.yield %14 : i32
// CHECK-NEXT:       }) : (!stream.stream<i32>, !stream.stream<i32>) -> !stream.stream<i32>
// CHECK-NEXT:       stream.yield %13 : !stream.stream<i32>
// CHECK-NEXT:     }) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>
// CHECK-NEXT:     func.return %3 : tensor<16x16xi32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
