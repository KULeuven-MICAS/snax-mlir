func.func @dynamic_matmul(%arg0: memref<?x?xi8>, %arg1: memref<?x?xi8, strided<[1, ?], offset:?>>, %arg2: memref<?x?xi32>) {
   %c0_i32 = arith.constant 0 : i32
   linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<?x?xi8>, memref<?x?xi8, strided<[1, ?], offset:?>>, i32, i32) outs(%arg2 : memref<?x?xi32>)
   return
}


transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.quantized_matmul">):
  // The actual tiling transformation takes tile sizes as attributes.
   %loop1, %loop2, %tiled = transform.structured.tile %arg1 [8, 8]
    : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
  transform.yield
}

