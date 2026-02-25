module {
  func.func public @streamer_add(%arg0: tensor<16xf64>, %arg1: tensor<16xf64>) -> tensor<16xf64> {
    %0 = tensor.empty() : tensor<16xf64>
    %1 = linalg.add ins(%arg0, %arg1 : tensor<16xf64>, tensor<16xf64>) outs(%0 : tensor<16xf64>) -> tensor<16xf64>
    %2 = linalg.sub ins(%arg0, %1 : tensor<16xf64>, tensor<16xf64>) outs(%0 : tensor<16xf64>) -> tensor<16xf64>
    %3 = linalg.mul ins(%arg0, %2 : tensor<16xf64>, tensor<16xf64>) outs(%0 : tensor<16xf64>) -> tensor<16xf64>
    return %3 : tensor<16xf64>
  }
}
