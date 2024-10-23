func.func @conv(%arg0 : tensor<1x18x18x16xi8>, %arg1 : tensor<16x3x3x16xi8>) -> tensor<1x16x16x16xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<1x16x16x16xi32>
  %conv = linalg.conv_2d_nhwc_fhwc_q ins(%arg0, %arg1, %c0_i32, %c0_i32 : tensor<1x18x18x16xi8>, tensor<16x3x3x16xi8>, i32, i32) outs(%0 : tensor<1x16x16x16xi32>) -> tensor<1x16x16x16xi32>
  func.return %conv : tensor<1x16x16x16xi32>
}
