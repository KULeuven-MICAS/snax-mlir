func.func @dynamic_matmul(%arg0: memref<?x?xi8>, %arg1: memref<?x?xi8, strided<[?, ?], offset:?>>, %arg2: memref<?x?xi32>) {
   %c0_i32 = arith.constant 0 : i32
   linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<?x?xi8>, memref<?x?xi8, strided<[?, ?], offset:?>>, i32, i32) outs(%arg2 : memref<?x?xi32>)
   return
}