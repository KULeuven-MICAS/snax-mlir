// RUN: snax-opt %s -p stream-bufferize --print-op-generic | filecheck %s
// RUN: snax-opt %s -p stream-bufferize,snax-bufferize --print-op-generic | filecheck %s --check-prefix FINAL

"func.func"() <{"sym_name" = "test", "function_type" = (tensor<16x16xi8>) -> tensor<16x16xi32>}> ({
^0(%arg0 : tensor<16x16xi8>):
  %empty = "tensor.empty"() : () -> tensor<16x16xi32>
  %0 = "stream.streaming_region"(%arg0, %arg0, %empty) <{"patterns" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>], "accelerator" = "snax_gemmx_stream", "operandSegmentSizes" = array<i32: 2, 1>}> ({
  ^0(%1 : !stream.stream<i8>, %2 : !stream.stream<i8>, %3 : !stream.stream<i32>):
    %4 = "stream.generic"(%1, %2) ({
    ^1(%in : i8, %in_1 : i8):
      %5 = "test.op"(%in, %in_1) : (i8, i8) -> i32
      stream.yield %5 : i32
    }) : (!stream.stream<i8>, !stream.stream<i8>) -> !stream.stream<i32>
    stream.yield %4 : !stream.stream<i32>
  }) : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>
  "func.return"(%0) : (tensor<16x16xi32>) -> ()
}) : () -> ()


// after this pass: bufferization ops should be inserted

// CHECK:      "func.func"() <{"sym_name" = "test", "function_type" = (tensor<16x16xi8>) -> tensor<16x16xi32>}> ({
// CHECK-NEXT: ^0(%arg0 : tensor<16x16xi8>):
// CHECK-NEXT:   %empty = "tensor.empty"() : () -> tensor<16x16xi32>
// CHECK-NEXT:   %0 = "bufferization.to_memref"(%arg0) : (tensor<16x16xi8>) -> memref<16x16xi8>
// CHECK-NEXT:   %1 = "bufferization.to_memref"(%empty) : (tensor<16x16xi32>) -> memref<16x16xi32>
// CHECK-NEXT:   "stream.streaming_region"(%0, %0, %1) <{"patterns" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>], "accelerator" = "snax_gemmx_stream", "operandSegmentSizes" = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^1(%2 : !stream.stream<i8>, %3 : !stream.stream<i8>, %4 : !stream.stream<i32>):
// CHECK-NEXT:     %5 = "stream.generic"(%2, %3) ({
// CHECK-NEXT:     ^2(%in : i8, %in_1 : i8):
// CHECK-NEXT:       %6 = "test.op"(%in, %in_1) : (i8, i8) -> i32
// CHECK-NEXT:       "stream.yield"(%6) : (i32) -> ()
// CHECK-NEXT:     }) : (!stream.stream<i8>, !stream.stream<i8>) -> !stream.stream<i32>
// CHECK-NEXT:     "stream.yield"(%5) : (!stream.stream<i32>) -> ()
// CHECK-NEXT:   }) : (memref<16x16xi8>, memref<16x16xi8>, memref<16x16xi32>) -> ()
// CHECK-NEXT:   %7 = "bufferization.to_tensor"(%1) <{"restrict"}> : (memref<16x16xi32>) -> tensor<16x16xi32>
// CHECK-NEXT:   "func.return"(%7) : (tensor<16x16xi32>) -> ()
// CHECK-NEXT: }) : () -> ()

// after --one-shot-bufferize: bufferization ops are gone, everything is memref, alloc has appeared

// FINAL:      "func.func"() <{"function_type" = (memref<16x16xi8>) -> memref<16x16xi32>, "sym_name" = "test"}> ({
// FINAL-NEXT: ^0(%arg0 : memref<16x16xi8>):
// FINAL-NEXT:   %0 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<16x16xi32>
// FINAL-NEXT:   "stream.streaming_region"(%arg0, %arg0, %0) <{"accelerator" = "snax_gemmx_stream", "operandSegmentSizes" = array<i32: 2, 1>, "patterns" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>]}> ({
// FINAL-NEXT:   ^1(%arg1 : !stream.stream<i8>, %arg2 : !stream.stream<i8>, %arg3 : !stream.stream<i32>):
// FINAL-NEXT:     %1 = "stream.generic"(%arg1, %arg2) ({
// FINAL-NEXT:     ^2(%arg4 : i8, %arg5 : i8):
// FINAL-NEXT:       %2 = "test.op"(%arg4, %arg5) : (i8, i8) -> i32
// FINAL-NEXT:       "stream.yield"(%2) : (i32) -> ()
// FINAL-NEXT:     }) : (!stream.stream<i8>, !stream.stream<i8>) -> !stream.stream<i32>
// FINAL-NEXT:     "stream.yield"(%1) : (!stream.stream<i32>) -> ()
// FINAL-NEXT:   }) : (memref<16x16xi8>, memref<16x16xi8>, memref<16x16xi32>) -> ()
// FINAL-NEXT:   "func.return"(%0) : (memref<16x16xi32>) -> ()
// FINAL-NEXT: }) : () -> ()
