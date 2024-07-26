func.func @streamer_matmul(%arg0: memref<?x?xi8>, %arg1: memref<?x?xi8, strided<[1, 16], offset:0>>, %arg2: memref<?x?xi32>) {
    %c0_i32 = arith.constant 0 : i32
    linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<?x?xi8>, memref<?x?xi8, strided<[1, 16], offset:0>>, i32, i32) outs(%arg2 : memref<?x?xi32>)
    return
}
