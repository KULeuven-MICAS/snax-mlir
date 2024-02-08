func.func @simple_matmul(%arg0 : memref<16x16xi8>, %arg1 : memref<16x16xi8>, %arg2 : memref<16x16xi32>) -> memref<16x16xi32> {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 0 : i32
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1, %0, %1 : memref<16x16xi8>, memref<16x16xi8>, i32, i32) outs(%arg2 : memref<16x16xi32>) {
    ^0(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
      %2 = arith.extsi %arg3 : i8 to i32
      %3 = arith.subi %2, %arg5 : i32
      %4 = arith.extsi %arg4 : i8 to i32
      %5 = arith.subi %4, %arg6 : i32
      %6 = arith.muli %3, %5 : i32
      %7 = arith.addi %arg7, %6 : i32
      linalg.yield %7 : i32
    }
    func.return %arg2 : memref<16x16xi32>
  }
