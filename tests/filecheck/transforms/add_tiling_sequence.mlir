// RUN: ./compiler/snax-opt --split-input-file %s -p add-tiling-sequence | filecheck %s

func.func @streamer_matmul(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16], offset:0>>, %arg2: memref<16x16xi32>) {
    %c0_i32 = arith.constant 0 : i32
    linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<16x16xi8>, memref<16x16xi8, strided<[1, 16], offset:0>>, i32, i32) outs(%arg2 : memref<16x16xi32>)
    return
}

//CHECK: builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul(%arg0 : memref<16x16xi8>, %arg1 : memref<16x16xi8, strided<[1, 16]>>, %arg2 : memref<16x16xi32>) {
//CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
//CHECK-NEXT:     linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, i32, i32) outs(%arg2 : memref<16x16xi32>)
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT:   "transform.sequence"() <{"failure_propagation_mode" = 1 : i32, "operandSegmentSizes" = array<i32: 0, 0>}> ({
//CHECK-NEXT:   ^0(%0 : !transform.any_op):
//CHECK-NEXT:     "transform.yield"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }
// -----

func.func @streamer_matmul(%arg0: memref<208x208xi8>, %arg1: memref<208x208xi8, strided<[1, 208], offset:0>>, %arg2: memref<208x208xi32>) {
    %c0_i32 = arith.constant 0 : i32
    linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<208x208xi8>, memref<208x208xi8, strided<[1, 208], offset:0>>, i32, i32) outs(%arg2 : memref<208x208xi32>)
    return
}

//CHECK: builtin.module {
//CHECK-NEXT:  func.func @streamer_matmul(%arg0 : memref<208x208xi8>, %arg1 : memref<208x208xi8, strided<[1, 208]>>, %arg2 : memref<208x208xi32>) {
//CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
//CHECK-NEXT:    linalg.quantized_matmul {"qmatmul_0"} ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<208x208xi8>, memref<208x208xi8, strided<[1, 208]>>, i32, i32) outs(%arg2 : memref<208x208xi32>)
//CHECK-NEXT:    func.return
//CHECK-NEXT:  }
//CHECK-NEXT:  "transform.sequence"() <{"failure_propagation_mode" = 1 : i32, "operandSegmentSizes" = array<i32: 0, 0>}> ({
//CHECK-NEXT:  ^0(%0 : !transform.any_op):
//CHECK-NEXT:    %1 = "transform.structured.match"(%0) <{"op_attrs" = {"qmatmul_0"}}> : (!transform.any_op) -> !transform.any_op
//CHECK-NEXT:    %2, %3, %4 = "transform.structured.tile_using_for"(%1) <{"static_sizes" = array<i64: 16, 0, 104>, "interchange" = array<i64: 1, 2, 0>, "scalable_sizes" = array<i1: false, false, false>}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
//CHECK-NEXT:    "transform.yield"() : () -> ()
//CHECK-NEXT:  }) : () -> ()
//CHECK-NEXT:}