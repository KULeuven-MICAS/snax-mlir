func.func @streamer_matmul(%arg0: memref<32x32xi8>, %arg1: memref<32x32xi8, strided<[1, 32], offset:0>>, %arg2: memref<32x32xi32>) {
    %c0_i32 = arith.constant 0 : i32
    linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<32x32xi8>, memref<32x32xi8, strided<[1, 32], offset:0>>, i32, i32) outs(%arg2 : memref<32x32xi32>)
    return
}
