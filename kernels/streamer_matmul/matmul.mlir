func.func @streamer_matmul(%arg0: memref<112x128xi8>, %arg1: memref<128x144xi8, strided<[1, 128], offset:0>>, %arg2: memref<112x144xi32>) {
    %c0_i32 = arith.constant 0 : i32
    linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<112x128xi8>, memref<128x144xi8, strided<[1, 128], offset:0>>, i32, i32) outs(%arg2 : memref<112x144xi32>)
    return
}
