func.func public @addition(%arg0 : tensor<16xi64>, %arg1 : tensor<16xi64>, %arg2 : tensor<16xi64>) -> tensor<16xi64>{
  %added = linalg.add ins(%arg0, %arg1 : tensor<16xi64>, tensor<16xi64>) outs(%arg2: tensor<16xi64>) -> tensor<16xi64>
  func.return %added : tensor<16xi64>
}
