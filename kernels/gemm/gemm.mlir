func.func @gemm(%arg0 : tensor<16x16xi8>, %arg1 : tensor<16x16xi8>, %arg2 : tensor<16x16xi32>) -> tensor<16x16xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<16x16xi32>
  %matmul = linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : tensor<16x16xi8>, tensor<16x16xi8>, i32, i32) outs(%0 : tensor<16x16xi32>) -> tensor<16x16xi32>
  %1 = tensor.empty() : tensor<16x16xi32>
  %out = linalg.add ins(%matmul, %arg2 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%1 : tensor<16x16xi32>) -> tensor<16x16xi32>
  func.return %out : tensor<16x16xi32>
}
