 func.func @streamer_matmul(%arg0: memref<128x128xi8>, %arg1: memref<128x128xi8, strided<[1, 128], offset:0>>, %arg2: memref<128x128xi32>) {
    %c0_i32 = arith.constant 0 : i32
    linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<128x128xi8>, memref<128x128xi8, strided<[1, 128], offset:0>>, i32, i32) outs(%arg2 : memref<128x128xi32>)
    return
 }


transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.quantized_matmul">):
  // The actual tiling transformation takes tile sizes as attributes.
   %loop1, %loop2, %tiled = transform.structured.tile %arg1 [32, 32]
    : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
  transform.yield
}




