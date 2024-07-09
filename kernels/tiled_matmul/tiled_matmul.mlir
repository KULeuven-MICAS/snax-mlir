func.func @simple_matmul(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16], offset:0>>, %arg2: memref<16x16xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %tile_size = arith.constant 8 : index
    scf.for %iv = %c0 to %c16 step %tile_size {
        scf.for %jv = %c0 to %c16 step %tile_size{
            scf.for %kv = %c0 to %c16 step %step_size{
                %tiled_A = "memref.subview" %arg0[%iv, %jv][%tile_size, %tile_size][%c1, %c1] //something after this
                %tiled_B = "memref.subview" %arg1[%iv, %jv][%tile_size, %tile_size][%c1, %c1]

                %tiled_D: "memref.subview" %arg1[%iv, %kv][%tile_size, %tile_size][%c1, %c1]
                
                %tiled_intermediate: memref<16x16xi8>
                linalg.quantized_matmul ins(%tiled_A, %tiled_B, %c0_i32, %c0_i32 : memref<8x8xi8>, memref<8x8xi8, strided<[1, 8], offset:0>>, i32, i32) outs(%tiled_intermediate : memref<8x8xi32>)

                %tiled_D = arith.add %tiled_D, %tiled_intermediate 

            }
        }
    }
    return
}
