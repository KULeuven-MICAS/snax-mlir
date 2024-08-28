// RUN: ./compiler/snax-opt --split-input-file %s -p add-tiling-sequence | filecheck %s

func.func @streamer_matmul(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32, strided<[1, 16], offset:0>>, %arg2: memref<16x16xf32>) {
    %c0_i32 = arith.constant 0 : f32
    linalg.matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<16x16xf32>, memref<16x16xf32, strided<[1, 16], offset:0>>, f32, f32) outs(%arg2 : memref<16x16xf32>)
    return
}
// Check that the sequence is added to the module, when nothing can be tiled
//CHECK:  builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul(%arg0 : memref<16x16xf32>, %arg1 : memref<16x16xf32, strided<[1, 16]>>, %arg2 : memref<16x16xf32>) {
//CHECK-NEXT:     %c0_i32 = arith.constant 0.000000e+00 : f32
//CHECK-NEXT:     linalg.matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<16x16xf32>, memref<16x16xf32, strided<[1, 16]>>, f32, f32) outs(%arg2 : memref<16x16xf32>)
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT:   "transform.sequence"() <{"failure_propagation_mode" = 2 : i32, "operandSegmentSizes" = array<i32: 0, 0>}> ({
//CHECK-NEXT:   ^0(%0 : !transform.any_op):
//CHECK-NEXT:     "transform.yield"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }
// -----

func.func @streamer_matmul(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32, strided<[1, 32], offset:0>>, %arg2: memref<32x32xf32>) {
    %c0_i32 = arith.constant 0 : f32
    linalg.matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<32x32xf32>, memref<32x32xf32, strided<[1, 32], offset:0>>, f32, f32) outs(%arg2 : memref<32x32xf32>)
    return
}

// Check that the sequence is added to the module, when tiling is possible
//CHECK: builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32, strided<[1, 32]>>, %arg2 : memref<32x32xf32>) {
//CHECK-NEXT:     %c0_i32 = arith.constant 0.000000e+00 : f32
//CHECK-NEXT:     linalg.matmul {"qmatmul_0"} ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<32x32xf32>, memref<32x32xf32, strided<[1, 32]>>, f32, f32) outs(%arg2 : memref<32x32xf32>)
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT:   "transform.sequence"() <{"failure_propagation_mode" = 2 : i32, "operandSegmentSizes" = array<i32: 0, 0>}> ({
//CHECK-NEXT:   ^0(%0 : !transform.any_op):
//CHECK-NEXT:     %1 = "transform.structured.match"(%0) <{"op_attrs" = {"qmatmul_0"}}> : (!transform.any_op) -> !transform.any_op
//CHECK-NEXT:     %2, %3, %4, %5 = "transform.structured.tile_using_for"(%1) <{"static_sizes" = array<i64: 16, 16, 16>, "interchange" = array<i64: 0, 2, 1>, "scalable_sizes" = array<i1: false, false, false>}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
//CHECK-NEXT:     "transform.yield"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }
// -----
