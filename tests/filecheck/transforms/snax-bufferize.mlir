// RUN: snax-opt %s -p snax-bufferize | filecheck %s
// TODO: add bufferization.clone to xDSL

"func.func"() <{"sym_name" = "test", "function_type" = (tensor<16x16xi8>) -> tensor<16x16xi32>}> ({
^bb0(%arg0 : tensor<16x16xi8>):
  %empty = "tensor.empty"() : () -> tensor<16x16xi32>
  %0 = "dart.operation"(%arg0, %arg0, %empty) <{"patterns" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>], "accelerator" = "snax_gemmx_stream", "operandSegmentSizes" = array<i32: 2, 1>}> ({
  ^bb0(%1 : !dart.stream<i8>, %2 : !dart.stream<i8>, %3 : !dart.stream<i32>):
    %4 = "dart.generic"(%1, %2) ({
    ^bb1(%in : i8, %in_1 : i8):
      %5 = "test.op"(%in, %in_1) : (i8, i8) -> i32
      dart.yield %5 : i32
    }) : (!dart.stream<i8>, !dart.stream<i8>) -> !dart.stream<i32>
    dart.yield %4 : !dart.stream<i32>
  }) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>
  "func.return"(%0) : (tensor<16x16xi32>) -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @test(%arg0 : memref<16x16xi8>) -> memref<16x16xi32> {
// CHECK-NEXT:     %0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi32>
// CHECK-NEXT:     "dart.operation"(%arg0, %arg0, %0) <{accelerator = "snax_gemmx_stream", operandSegmentSizes = array<i32: 2, 1>, patterns = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>]}> ({
// CHECK-NEXT:     ^bb0(%arg1 : !dart.stream<i8>, %arg2 : !dart.stream<i8>, %arg3 : !dart.stream<i32>):
// CHECK-NEXT:       %1 = "dart.generic"(%arg1, %arg2) ({
// CHECK-NEXT:       ^bb1(%arg4 : i8, %arg5 : i8):
// CHECK-NEXT:         %2 = "test.op"(%arg4, %arg5) : (i8, i8) -> i32
// CHECK-NEXT:         dart.yield %2 : i32
// CHECK-NEXT:       }) : (!dart.stream<i8>, !dart.stream<i8>) -> !dart.stream<i32>
// CHECK-NEXT:       dart.yield %1 : !dart.stream<i32>
// CHECK-NEXT:     }) : (memref<16x16xi8>, memref<16x16xi8>, memref<16x16xi32>) -> ()
// CHECK-NEXT:     func.return %0 : memref<16x16xi32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
