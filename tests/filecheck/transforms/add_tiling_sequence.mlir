// RUN: ./compiler/snax-opt --split-input-file %s -p add-tiling-sequence | filecheck %s

func.func @streamer_matmul(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32, strided<[1, 16], offset:0>>, %arg2: memref<16x16xf32>) {
    %c0_i32 = arith.constant 0 : f32
    linalg.matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<16x16xf32>, memref<16x16xf32, strided<[1, 16], offset:0>>, f32, f32) outs(%arg2 : memref<16x16xf32>)
    return
}


