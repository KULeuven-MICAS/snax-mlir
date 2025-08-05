func.func public @rescale_down(%arg0: tensor<64xi32>) -> tensor<64xi8> {
  %0 = tosa.rescale %arg0 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1140768826>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 48>} : (tensor<64xi32>) -> tensor<64xi8>
  return %0 : tensor<64xi8>
}
