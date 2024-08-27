func.func @streamer_matmul(%arg0: memref<40x56xi8>, %arg1: memref<56x24xi8, strided<[1, 56], offset:0>>, %arg2: memref<40x24xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c40 = arith.constant 40 : index
  %c56 = arith.constant 56 : index
  %short_tile_size = arith.constant 8 : index
  %long_tile_size = arith.constant 16 : index

  scf.for %iv = %c0 to %c56 step %short_tile_size {
    scf.for %jv = %c0 to %c40 step %short_tile_size {
      // Extract tiles from input matrices
      %tiled_A = memref.subview %arg0[%jv, %iv][%short_tile_size, %short_tile_size][%c1, %c1] : memref<40x56xi8> to memref<?x?xi8, strided<[56, 1], offset: ?>>
      %tiled_B = memref.subview %arg1[%iv, %c0][%short_tile_size, %long_tile_size][%c1, %c1] : memref<56x24xi8, strided<[1, 56], offset:0>> to memref<?x?xi8, strided<[1, 56], offset: ?>>
      %tiled_D = memref.subview %arg2[%jv, %c0][%short_tile_size, %long_tile_size][%c1, %c1] : memref<40x24xi32> to memref<?x?xi32, strided<[24, 1], offset: ?>>
      
      // Perform the quantized matrix multiplication on the tiles
      linalg.quantized_matmul ins(%tiled_A, %tiled_B, %c0_i32, %c0_i32 : memref<?x?xi8, strided<[56, 1], offset: ?>>, memref<?x?xi8, strided<[1, 56], offset: ?>>, i32, i32) outs(%tiled_D : memref<?x?xi32, strided<[24, 1], offset: ?>>)
      
    }
  }
  
  return
}
