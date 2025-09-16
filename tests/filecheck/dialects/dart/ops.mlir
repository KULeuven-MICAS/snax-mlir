// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_SINGLETRIP

%s0, %s1, %s2 = "test.op"() : () -> (!dart.stream<i8>, !dart.stream<i32>, !dart.stream<f32>)
%t0, %t1 = "test.op"() : () -> (tensor<16x16xi8>, tensor<16x16xi32>)

%0 = "dart.operation"(%t0, %t0, %t1) <{patterns = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>], accelerator = "snax_gemmx_stream", operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%1 : !dart.stream<i8>, %2 : !dart.stream<i8>, %3 : !dart.stream<i32>):
  %4 = "dart.generic"(%1, %2) ({
  ^bb1(%in : i8, %in_1 : i8):
    %5 = "test.op"(%in, %in_1) : (i8, i8) -> i32
    dart.yield %5 : i32
  }) : (!dart.stream<i8>, !dart.stream<i8>) -> !dart.stream<i32>
  dart.yield %4 : !dart.stream<i32>
}) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>


// CHECK: builtin.module {
// CHECK-NEXT:   %s0, %s1, %s2 = "test.op"() : () -> (!dart.stream<i8>, !dart.stream<i32>, !dart.stream<f32>)
// CHECK-NEXT:   %t0, %t1 = "test.op"() : () -> (tensor<16x16xi8>, tensor<16x16xi32>)
// CHECK-NEXT:   %0 = "dart.operation"(%t0, %t0, %t1) <{patterns = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>], accelerator = "snax_gemmx_stream", operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%1 : !dart.stream<i8>, %2 : !dart.stream<i8>, %3 : !dart.stream<i32>):
// CHECK-NEXT:     %4 = "dart.generic"(%1, %2) ({
// CHECK-NEXT:     ^bb1(%in : i8, %in_1 : i8):
// CHECK-NEXT:       %5 = "test.op"(%in, %in_1) : (i8, i8) -> i32
// CHECK-NEXT:       dart.yield %5 : i32
// CHECK-NEXT:     }) : (!dart.stream<i8>, !dart.stream<i8>) -> !dart.stream<i32>
// CHECK-NEXT:     dart.yield %4 : !dart.stream<i32>
// CHECK-NEXT:   }) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>
// CHECK-NEXT: }

// -----


%s0, %s1, %s2 = "test.op"() : () -> (!dart.stream<i8>, !dart.stream<i32>, !dart.stream<f32>)
%t0, %t1 = "test.op"() : () -> (tensor<16x16xi8>, tensor<16x16xi32>)

%0 = "dart.schedule"(%t0, %t0, %t1) <{bounds = [1: index, 2: index, 3: index], tiles = [[1 : index, 3: index], [5: index, 7: index]], patterns = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>], accelerator = "snax_gemmx_stream", operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%1 : !dart.stream<i8>, %2 : !dart.stream<i8>, %3 : !dart.stream<i32>):
  %4 = "dart.generic"(%1, %2) ({
  ^bb1(%in : i8, %in_1 : i8):
    %5 = "test.op"(%in, %in_1) : (i8, i8) -> i32
    dart.yield %5 : i32
  }) : (!dart.stream<i8>, !dart.stream<i8>) -> !dart.stream<i32>
  dart.yield %4 : !dart.stream<i32>
}) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>

 // CHECK: builtin.module {
 // CHECK-NEXT:   %s0, %s1, %s2 = "test.op"() : () -> (!dart.stream<i8>, !dart.stream<i32>, !dart.stream<f32>)
 // CHECK-NEXT:   %t0, %t1 = "test.op"() : () -> (tensor<16x16xi8>, tensor<16x16xi32>)
 // CHECK-NEXT:   %0 = "dart.schedule"(%t0, %t0, %t1) <{bounds = [1 : index, 2 : index, 3 : index], tiles = [[1 : index, 3 : index], [5 : index, 7 : index]], patterns = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>], accelerator = "snax_gemmx_stream", operandSegmentSizes = array<i32: 2, 1>}> ({
 // CHECK-NEXT:   ^bb0(%1 : !dart.stream<i8>, %2 : !dart.stream<i8>, %3 : !dart.stream<i32>):
 // CHECK-NEXT:     %4 = "dart.generic"(%1, %2) ({
 // CHECK-NEXT:     ^bb1(%in : i8, %in_1 : i8):
 // CHECK-NEXT:       %5 = "test.op"(%in, %in_1) : (i8, i8) -> i32
 // CHECK-NEXT:       dart.yield %5 : i32
 // CHECK-NEXT:     }) : (!dart.stream<i8>, !dart.stream<i8>) -> !dart.stream<i32>
 // CHECK-NEXT:     dart.yield %4 : !dart.stream<i32>
 // CHECK-NEXT:   }) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>
 // CHECK-NEXT: }
