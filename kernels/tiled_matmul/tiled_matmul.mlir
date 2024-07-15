func.func @tiled_matmul(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16], offset:0>>, %arg2: memref<16x16xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %tile_size = arith.constant 8 : index

  scf.for %iv = %c0 to %c16 step %tile_size {
    scf.for %jv = %c0 to %c16 step %tile_size {
      scf.for %kv = %c0 to %c16 step %tile_size {
        // Extract tiles from input matrices
        %tiled_A = memref.subview %arg0[%iv, %jv][%tile_size, %tile_size][%c1, %c1] : memref<16x16xi8> to memref<?x?xi8, strided<[16, 1], offset: ?>>
        %tiled_B = memref.subview %arg1[%kv, %jv][%tile_size, %tile_size][%c1, %c1] : memref<16x16xi8, strided<[1, 16], offset:0>> to memref<?x?xi8, strided<[1, 16], offset: ?>>
        %tiled_D = memref.subview %arg2[%iv, %kv][%tile_size, %tile_size][%c1, %c1] : memref<16x16xi32> to memref<?x?xi32, strided<[16, 1], offset: ?>>
        
        %tiled_intermediate = memref.alloc() : memref<8x8xi32, strided<[16, 1], offset: 1>>
        %tiled_added = memref.alloc() : memref<8x8xi32, strided<[16, 1], offset: 1>>

        // Perform the quantized matrix multiplication on the tiles
        linalg.quantized_matmul ins(%tiled_A, %tiled_B, %c0_i32, %c0_i32 : memref<?x?xi8, strided<[16, 1], offset: ?>>, memref<?x?xi8, strided<[1, 16], offset: ?>>, i32, i32) outs(%tiled_intermediate : memref<8x8xi32, strided<[16, 1], offset: 1>>)
        %tiled_D_8 = memref.cast %tiled_D : memref<?x?xi32, strided<[16, 1], offset: ?>> to memref<8x8xi32, strided<[16, 1], offset: 1>>
        linalg.add ins(%tiled_D_8, %tiled_intermediate : memref<8x8xi32, strided<[16, 1], offset: 1>>, memref<8x8xi32, strided<[16, 1], offset: 1>>)
             outs(%tiled_added : memref<8x8xi32, strided<[16, 1], offset: 1>>)
        
        %tiled_added_D = memref.cast %tiled_added : memref<8x8xi32, strided<[16, 1], offset: 1>> to memref<?x?xi32, strided<[16, 1], offset: ?>>
        memref.copy %tiled_added_D, %tiled_D :  memref<?x?xi32, strided<[16, 1], offset: ?>> to  memref<?x?xi32, strided<[16, 1], offset: ?>>
      }
    }
  }
  
  return
}


// func.func @tiled_matmul(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16], offset:0>>, %arg2: memref<16x16xi32>) {
//   %c0_i32 = arith.constant 0 : i32
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c16 = arith.constant 16 : index
//   %tile_size = arith.constant 8 : index

//   scf.for %iv = %c0 to %c16 step %tile_size {
//     scf.for %jv = %c0 to %c16 step %tile_size {
//       scf.for %kv = %c0 to %c16 step %tile_size {
//         // Extract tiles from input matrices
//         %tiled_A = memref.subview %arg0[%iv, %jv][%tile_size, %tile_size][%c1, %c1] : memref<16x16xi8> to memref<?x?xi8, strided<[16, 1], offset: ?>>
//         %tiled_B = memref.subview %arg1[%kv, %jv][%tile_size, %tile_size][%c1, %c1] : memref<16x16xi8, strided<[1, 16], offset:0>> to memref<?x?xi8, strided<[1, 16], offset: ?>>
//         %tiled_D = memref.subview %arg2[%iv, %kv][%tile_size, %tile_size][%c1, %c1] : memref<16x16xi32> to memref<?x?xi32, strided<[16, 1], offset: ?>>
        
//         %tiled_intermediate = memref.alloc() : memref<?x?xi32, strided<[16, 1], offset: ?>>
//         // Perform the quantized matrix multiplication on the tiles
//         linalg.quantized_matmul ins(%tiled_A, %tiled_B, %c0_i32, %c0_i32 : memref<?x?xi8, strided<[16, 1], offset: ?>>, memref<?x?xi8, strided<[1, 16], offset: ?>>, i32, i32) outs(%tiled_D : memref<?x?xi32, strided<[16, 1], offset: ?>>)
//       }
//     }
//   }
  
//   return
// }
