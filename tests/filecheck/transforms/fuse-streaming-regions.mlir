// RUN: snax-opt --split-input-file -p dart-fuse-operations %s | filecheck %s

func.func @streamer_matmul(%arg0 : tensor<16x16xi8>, %arg1 : tensor<16x16xi8>, %arg2 : tensor<16x16xi32>) -> tensor<16x16xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<16x16xi32>

  // first streaming region (matmul):
  %1 = "dart.operation"(%arg0, %arg1, %0) <{"patterns" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>], "accelerator" = "snax_gemmx", "operandSegmentSizes" = array<i32: 2, 1>}> ({
  ^bb0(%2 : !dart.stream<i8>, %3 : !dart.stream<i8>, %4 : !dart.stream<i32>):
    %5 = "dart.generic"(%2, %3, %c0_i32, %c0_i32) <{"library_call" = "snax_gemmx"}> ({
    ^bb1(%in : i8, %in_1 : i8, %in_2 : i32, %in_3 : i32, %out : i32):
      %6 = kernel.qmac %in, %in_1 zp_lhs : %in_2 zp_rhs : %in_3 : i8, i8, i32, i32 -> i32
      dart.yield %6 : i32
    }) : (!dart.stream<i8>, !dart.stream<i8>, i32, i32) -> !dart.stream<i32>
    dart.yield %5 : !dart.stream<i32>
  }) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>

  %7 = tensor.empty() : tensor<16x16xi32>

  // second streaming region (elementwise add):
  %8 = "dart.operation"(%1, %arg2, %7) <{"patterns" = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], "accelerator" = "snax_gemmx", "operandSegmentSizes" = array<i32: 2, 1>}> ({
  ^bb2(%9 : !dart.stream<i32>, %10 : !dart.stream<i32>, %11 : !dart.stream<i32>):
    %12 = "dart.generic"(%9, %10) <{"library_call" = "snax_gemmx"}> ({
    ^bb3(%in_4 : i32, %in_5 : i32, %out_1 : i32):
      %13 = kernel.add %in_4, %in_5 : i32, i32 -> i32
      dart.yield %13 : i32
    }) : (!dart.stream<i32>, !dart.stream<i32>) -> !dart.stream<i32>
    dart.yield %12 : !dart.stream<i32>
  }) : (tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>

  %13 = tensor.empty() : tensor<16x16xi32>

  // third streaming region (elementwise add):
  %14 = "dart.operation"(%8, %arg2, %13) <{"patterns" = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], "accelerator" = "snax_gemmx", "operandSegmentSizes" = array<i32: 2, 1>}> ({
  ^bb2(%15 : !dart.stream<i32>, %16 : !dart.stream<i32>, %17 : !dart.stream<i32>):
    %18 = "dart.generic"(%15, %16) <{"library_call" = "snax_gemmx"}> ({
    ^bb3(%in_6 : i32, %in_7 : i32, %out_2 : i32):
      %19 = kernel.add %in_6, %in_7 : i32, i32 -> i32
      dart.yield %19 : i32
    }) : (!dart.stream<i32>, !dart.stream<i32>) -> !dart.stream<i32>
    dart.yield %18 : !dart.stream<i32>
  }) : (tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>

  func.return %14 : tensor<16x16xi32>
}


// CHECK: builtin.module {
// CHECK-NEXT:   func.func @streamer_matmul(%arg0 : tensor<16x16xi8>, %arg1 : tensor<16x16xi8>, %arg2 : tensor<16x16xi32>) -> tensor<16x16xi32> {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = tensor.empty() : tensor<16x16xi32>
// CHECK-NEXT:     %1 = tensor.empty() : tensor<16x16xi32>
// CHECK-NEXT:     %2 = tensor.empty() : tensor<16x16xi32>
// CHECK-NEXT:     %3 = "dart.operation"(%arg0, %arg1, %arg2, %arg2, %2) <{patterns = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], accelerator = "snax_gemmx", operandSegmentSizes = array<i32: 4, 1>}> ({
// CHECK-NEXT:     ^bb0(%4 : !dart.stream<i8>, %5 : !dart.stream<i8>, %6 : !dart.stream<i32>, %7 : !dart.stream<i32>, %8 : !dart.stream<i32>):
// CHECK-NEXT:       %9 = "dart.generic"(%4, %5, %c0_i32, %c0_i32) <{library_call = "snax_gemmx"}> ({
// CHECK-NEXT:       ^bb1(%in : i8, %in_1 : i8, %in_2 : i32, %in_3 : i32, %out : i32):
// CHECK-NEXT:         %10 = kernel.qmac %in, %in_1 zp_lhs : %in_2 zp_rhs : %in_3 : i8, i8, i32, i32 -> i32
// CHECK-NEXT:         dart.yield %10 : i32
// CHECK-NEXT:       }) : (!dart.stream<i8>, !dart.stream<i8>, i32, i32) -> !dart.stream<i32>
// CHECK-NEXT:       %11 = "dart.generic"(%9, %6) <{library_call = "snax_gemmx"}> ({
// CHECK-NEXT:       ^bb2(%in_4 : i32, %in_5 : i32, %out_1 : i32):
// CHECK-NEXT:         %12 = kernel.add %in_4, %in_5 : i32, i32 -> i32
// CHECK-NEXT:         dart.yield %12 : i32
// CHECK-NEXT:       }) : (!dart.stream<i32>, !dart.stream<i32>) -> !dart.stream<i32>
// CHECK-NEXT:       %13 = "dart.generic"(%11, %7) <{library_call = "snax_gemmx"}> ({
// CHECK-NEXT:       ^bb3(%in_6 : i32, %in_7 : i32, %out_2 : i32):
// CHECK-NEXT:         %14 = kernel.add %in_6, %in_7 : i32, i32 -> i32
// CHECK-NEXT:         dart.yield %14 : i32
// CHECK-NEXT:       }) : (!dart.stream<i32>, !dart.stream<i32>) -> !dart.stream<i32>
// CHECK-NEXT:       dart.yield %13 : !dart.stream<i32>
// CHECK-NEXT:     }) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>
// CHECK-NEXT:     func.return %3 : tensor<16x16xi32>
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----


func.func @streamer_matmul(%arg0 : tensor<16x16xi8>, %arg1 : tensor<16x16xi8>, %arg2 : tensor<16xi32>) -> tensor<16x16xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<16x16xi32>

  // first streaming region (matmul):
  %1 = "dart.operation"(%arg0, %arg1, %0) <{"patterns" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>], "accelerator" = "snax_gemmx", "operandSegmentSizes" = array<i32: 2, 1>}> ({
  ^bb0(%2 : !dart.stream<i8>, %3 : !dart.stream<i8>, %4 : !dart.stream<i32>):
    %5 = "dart.generic"(%2, %3, %c0_i32, %c0_i32) <{"library_call" = "snax_gemmx"}> ({
    ^bb1(%in : i8, %in_1 : i8, %in_2 : i32, %in_3 : i32, %out : i32):
      %6 = kernel.qmac %in, %in_1 zp_lhs : %in_2 zp_rhs : %in_3 : i8, i8, i32, i32 -> i32
      dart.yield %6 : i32
    }) : (!dart.stream<i8>, !dart.stream<i8>, i32, i32) -> !dart.stream<i32>
    dart.yield %5 : !dart.stream<i32>
  }) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>

  %7 = tensor.empty() : tensor<16x16xi32>

  // second streaming region (broadcasted elementwise add):
  %14 = "dart.operation"(%1, %arg2, %7) <{"patterns" = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], "accelerator" = "snax_gemmx", "operandSegmentSizes" = array<i32: 2, 1>}> ({
  ^bb2(%15 : !dart.stream<i32>, %16 : !dart.stream<i32>, %17 : !dart.stream<i32>):
    %18 = "dart.generic"(%15, %16) <{"library_call" = "snax_gemmx"}> ({
    ^bb3(%in_6 : i32, %in_7 : i32, %out_2 : i32):
      %19 = kernel.add %in_6, %in_7 : i32, i32 -> i32
      dart.yield %19 : i32
    }) : (!dart.stream<i32>, !dart.stream<i32>) -> !dart.stream<i32>
    dart.yield %18 : !dart.stream<i32>
  }) : (tensor<16x16xi32>, tensor<16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>

  func.return %14 : tensor<16x16xi32>
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @streamer_matmul(%arg0 : tensor<16x16xi8>, %arg1 : tensor<16x16xi8>, %arg2 : tensor<16xi32>) -> tensor<16x16xi32> {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = tensor.empty() : tensor<16x16xi32>
// CHECK-NEXT:     %1 = tensor.empty() : tensor<16x16xi32>
// CHECK-NEXT:     %2 = "dart.operation"(%arg0, %arg1, %arg2, %1) <{patterns = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], accelerator = "snax_gemmx", operandSegmentSizes = array<i32: 3, 1>}> ({
// CHECK-NEXT:     ^bb0(%3 : !dart.stream<i8>, %4 : !dart.stream<i8>, %5 : !dart.stream<i32>, %6 : !dart.stream<i32>):
// CHECK-NEXT:       %7 = "dart.generic"(%3, %4, %c0_i32, %c0_i32) <{library_call = "snax_gemmx"}> ({
// CHECK-NEXT:       ^bb1(%in : i8, %in_1 : i8, %in_2 : i32, %in_3 : i32, %out : i32):
// CHECK-NEXT:         %8 = kernel.qmac %in, %in_1 zp_lhs : %in_2 zp_rhs : %in_3 : i8, i8, i32, i32 -> i32
// CHECK-NEXT:         dart.yield %8 : i32
// CHECK-NEXT:       }) : (!dart.stream<i8>, !dart.stream<i8>, i32, i32) -> !dart.stream<i32>
// CHECK-NEXT:       %9 = "dart.generic"(%7, %5) <{library_call = "snax_gemmx"}> ({
// CHECK-NEXT:       ^bb2(%in_4 : i32, %in_5 : i32, %out_1 : i32):
// CHECK-NEXT:         %10 = kernel.add %in_4, %in_5 : i32, i32 -> i32
// CHECK-NEXT:         dart.yield %10 : i32
// CHECK-NEXT:       }) : (!dart.stream<i32>, !dart.stream<i32>) -> !dart.stream<i32>
// CHECK-NEXT:       dart.yield %9 : !dart.stream<i32>
// CHECK-NEXT:     }) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>
// CHECK-NEXT:     func.return %2 : tensor<16x16xi32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
