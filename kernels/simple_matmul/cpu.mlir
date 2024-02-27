func.func public @simple_matmul(%A: memref<16x16xi8, 1 : i32>,
                                %B: memref<16x32xi8, 1 : i32>,
                                %C: memref<16x32xi32, 1 : i32>) -> () {
  func.call @simple_matmul_cpu(%A, %B, %C) : (memref<16x16xi8, 1 : i32>, memref<16x32xi8, 1 : i32>, memref<16x32xi32, 1 : i32>) -> ()
  return
} 
func.func private @simple_matmul_cpu(%A : memref<16x16xi8, 1 : i32>, %B : memref<16x32xi8, 1 : i32>, %C : memref<16x32xi32, 1 : i32>)
