// RUN: snax-opt --split-input-file -p convert-linalg-to-dart %s | filecheck %s

%arg0, %arg1, %arg2 = "test.op"() : () -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>)
%c0_i32 = arith.constant 0 : i32
// CHECK: builtin.module {
// CHECK-NEXT:  %arg0, %arg1, %arg2 = "test.op"() : () -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>)
// CHECK-NEXT:  %c0_i32 = arith.constant 0 : i32

%0 = tensor.empty() : tensor<16x16xi32>
// CHECK-NEXT:  %0 = tensor.empty() : tensor<16x16xi32>

%1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], library_call = "snax_gemmx_stream"} ins(%arg0, %arg1, %c0_i32, %c0_i32 : tensor<16x16xi8>, tensor<16x16xi8>, i32, i32) outs(%0 : tensor<16x16xi32>) {
^bb0(%in : i8, %in_1 : i8, %in_2 : i32, %in_3 : i32, %out : i32):
  %2 = kernel.qmac %in, %in_1 zp_lhs : %in_2 zp_rhs : %in_3 : i8, i8, i32, i32 -> i32
  linalg.yield %2 : i32
} -> tensor<16x16xi32>
// CHECK-NEXT: %1 = "dart.operation"(%arg0, %arg1, %0) <{patterns = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], accelerator = "snax_gemmx", operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT: ^bb0(%2 : !dart.stream<i8>, %3 : !dart.stream<i8>, %4 : !dart.stream<i32>):
// CHECK-NEXT:   %5 = "dart.generic"(%2, %3, %c0_i32, %c0_i32) <{library_call = "snax_gemmx"}> ({
// CHECK-NEXT:   ^bb1(%in : i8, %in_1 : i8, %in_2 : i32, %in_3 : i32, %out : i32):
// CHECK-NEXT:     %6 = kernel.qmac %in, %in_1 zp_lhs : %in_2 zp_rhs : %in_3 : i8, i8, i32, i32 -> i32
// CHECK-NEXT:     dart.yield %6 : i32
// CHECK-NEXT:   }) : (!dart.stream<i8>, !dart.stream<i8>, i32, i32) -> !dart.stream<i32>
// CHECK-NEXT:   dart.yield %5 : !dart.stream<i32>
// CHECK-NEXT: }) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>

%3 = tensor.empty() : tensor<16x16xi32>
//CHECK-NEXT:   %7 = tensor.empty() : tensor<16x16xi32>

%4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"], library_call = "snax_gemmx_stream"} ins(%1, %arg2 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%3 : tensor<16x16xi32>) {
^bb1(%in_4 : i32, %in_5 : i32, %out_1 : i32):
  %5 = kernel.add %in_4, %in_5 : i32, i32 -> i32
  linalg.yield %5 : i32
} -> tensor<16x16xi32>

//CHECK-NEXT: %8 = "dart.operation"(%1, %arg2, %7) <{patterns = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], accelerator = "snax_gemmx", operandSegmentSizes = array<i32: 2, 1>}> ({
//CHECK-NEXT: ^bb2(%9 : !dart.stream<i32>, %10 : !dart.stream<i32>, %11 : !dart.stream<i32>):
//CHECK-NEXT:   %12 = "dart.generic"(%9, %10) <{library_call = "snax_gemmx"}> ({
//CHECK-NEXT:   ^bb3(%in_4 : i32, %in_5 : i32, %out_1 : i32):
//CHECK-NEXT:     %13 = kernel.add %in_4, %in_5 : i32, i32 -> i32
//CHECK-NEXT:     dart.yield %13 : i32
//CHECK-NEXT:   }) : (!dart.stream<i32>, !dart.stream<i32>) -> !dart.stream<i32>
//CHECK-NEXT:   dart.yield %12 : !dart.stream<i32>
//CHECK-NEXT: }) : (tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>

%6 = arith.constant dense<5> : tensor<16x16xi32>
//CHECK-NEXT:  %14 = arith.constant dense<5> : tensor<16x16xi32>

%7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], library_call = "snax_gemmx_stream"} ins(%arg0, %arg1, %c0_i32, %c0_i32 : tensor<16x16xi8>, tensor<16x16xi8>, i32, i32) outs(%6 : tensor<16x16xi32>) {
^bb0(%in_6 : i8, %in_7 : i8, %in_8 : i32, %in_9 : i32, %out_2 : i32):
  %8 = kernel.qmac %in_6, %in_7 zp_lhs : %in_8 zp_rhs : %in_9 : i8, i8, i32, i32 -> i32
  linalg.yield %8 : i32
} -> tensor<16x16xi32>

//CHECK-NEXT:  %15 = tensor.empty() : tensor<16x16xi32>
//CHECK-NEXT:  %16 = "dart.operation"(%arg0, %arg1, %15) <{patterns = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], accelerator = "snax_gemmx", operandSegmentSizes = array<i32: 2, 1>}> ({
//CHECK-NEXT:  ^bb4(%17 : !dart.stream<i8>, %18 : !dart.stream<i8>, %19 : !dart.stream<i32>):
//CHECK-NEXT:    %20 = "dart.generic"(%17, %18, %c0_i32, %c0_i32) <{library_call = "snax_gemmx"}> ({
//CHECK-NEXT:    ^bb5(%in_6 : i8, %in_7 : i8, %in_8 : i32, %in_9 : i32, %out_2 : i32):
//CHECK-NEXT:      %21 = kernel.qmac %in_6, %in_7 zp_lhs : %in_8 zp_rhs : %in_9 : i8, i8, i32, i32 -> i32
//CHECK-NEXT:      dart.yield %21 : i32
//CHECK-NEXT:    }) : (!dart.stream<i8>, !dart.stream<i8>, i32, i32) -> !dart.stream<i32>
//CHECK-NEXT:    dart.yield %20 : !dart.stream<i32>
//CHECK-NEXT:  }) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>
//CHECK-NEXT:  %22 = tensor.empty() : tensor<16x16xi32>
//CHECK-NEXT:  %23 = "dart.operation"(%16, %14, %22) <{patterns = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], accelerator = "snax_gemmx", operandSegmentSizes = array<i32: 2, 1>}> ({
//CHECK-NEXT:  ^bb6(%24 : !dart.stream<i32>, %25 : !dart.stream<i32>, %26 : !dart.stream<i32>):
//CHECK-NEXT:    %27 = "dart.generic"(%24, %25) <{library_call = "snax_gemmx"}> ({
//CHECK-NEXT:    ^bb7(%28 : i32, %29 : i32, %30 : i32):
//CHECK-NEXT:      %31 = kernel.add %28, %29 : i32, i32 -> i32
//CHECK-NEXT:      dart.yield %31 : i32
//CHECK-NEXT:    }) : (!dart.stream<i32>, !dart.stream<i32>) -> !dart.stream<i32>
//CHECK-NEXT:    dart.yield %27 : !dart.stream<i32>
//CHECK-NEXT:  }) : (tensor<16x16xi32>, tensor<16x16xi32>, tensor<16x16xi32>) -> tensor

%9 = tensor.empty() : tensor<16x16xi32>
%10 = arith.constant dense<5> : tensor<16xi32>
%11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<16xi32>) outs(%9 : tensor<16x16xi32>) {
^bb0(%arg3 : i32, %arg4 : i32):
  linalg.yield %arg3 : i32
} -> tensor<16x16xi32>

// CHECK-NEXT: %32 = tensor.empty() : tensor<16x16xi32>
// CHECK-NEXT: %33 = arith.constant dense<5> : tensor<16xi32>
// CHECK-NEXT: %34 = tensor.empty() : tensor<16x16xi32>

%12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], library_call = "snax_gemmx_stream"} ins(%arg0, %arg1, %c0_i32, %c0_i32 : tensor<16x16xi8>, tensor<16x16xi8>, i32, i32) outs(%11 : tensor<16x16xi32>) {
^bb0(%in_10 : i8, %in_11 : i8, %in_12 : i32, %in_13 : i32, %out_3 : i32):
  %13 = kernel.qmac %in_10, %in_11 zp_lhs : %in_12 zp_rhs : %in_13 : i8, i8, i32, i32 -> i32
  linalg.yield %13 : i32
} -> tensor<16x16xi32>

// CHECK-NEXT: %35 = "dart.operation"(%arg0, %arg1, %34) <{patterns = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], accelerator = "snax_gemmx", operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT: ^bb8(%36 : !dart.stream<i8>, %37 : !dart.stream<i8>, %38 : !dart.stream<i32>):
// CHECK-NEXT:   %39 = "dart.generic"(%36, %37, %c0_i32, %c0_i32) <{library_call = "snax_gemmx"}> ({
// CHECK-NEXT:   ^bb9(%in_10 : i8, %in_11 : i8, %in_12 : i32, %in_13 : i32, %out_3 : i32):
// CHECK-NEXT:     %40 = kernel.qmac %in_10, %in_11 zp_lhs : %in_12 zp_rhs : %in_13 : i8, i8, i32, i32 -> i32
// CHECK-NEXT:     dart.yield %40 : i32
// CHECK-NEXT:   }) : (!dart.stream<i8>, !dart.stream<i8>, i32, i32) -> !dart.stream<i32>
// CHECK-NEXT:   dart.yield %39 : !dart.stream<i32>
// CHECK-NEXT: }) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>
// CHECK-NEXT: %41 = "dart.operation"(%35, %33, %32) <{patterns = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], accelerator = "snax_gemmx", operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT: ^bb10(%42 : !dart.stream<i32>, %43 : !dart.stream<i32>, %44 : !dart.stream<i32>):
// CHECK-NEXT:   %45 = "dart.generic"(%42, %43) <{library_call = "snax_gemmx"}> ({
// CHECK-NEXT:   ^bb11(%46 : i32, %47 : i32, %48 : i32):
// CHECK-NEXT:     %49 = kernel.add %46, %47 : i32, i32 -> i32
// CHECK-NEXT:     dart.yield %49 : i32
// CHECK-NEXT:   }) : (!dart.stream<i32>, !dart.stream<i32>) -> !dart.stream<i32>
// CHECK-NEXT:   dart.yield %45 : !dart.stream<i32>
// CHECK-NEXT: }) : (tensor<16x16xi32>, tensor<16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>
