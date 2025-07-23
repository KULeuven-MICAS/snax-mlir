  func.func public @rescale_combined(%arg0: tensor<64xi8>) -> tensor<64xi32> {
    %0 = tosa.rescale %arg0 {double_round = true, input_zp = 38 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 10>} : (tensor<64xi8>) -> tensor<64xi32>
    %1 = tosa.rescale %0 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1657902019>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 33>} : (tensor<64xi32>) -> tensor<64xi32>
    return %0 : tensor<64xi32>
  }
