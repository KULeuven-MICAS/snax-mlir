// func.func public @rescale_down(%arg0: tensor<1x114x114x64xi8>) -> tensor<1x56x56x64xi8> {
//   %0 = "tosa.max_pool2d"(%arg0) <{kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<1x114x114x64xi8>) -> tensor<1x56x56x64xi8>
//   return %0 : tensor<1x56x56x64xi8>
// }

func.func public @rescale_down(%arg0: tensor<1x28x28x64xi8>) -> tensor<1x13x13x64xi8> {
  %0 = "tosa.max_pool2d"(%arg0) <{kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<1x28x28x64xi8>) -> tensor<1x13x13x64xi8>
  return %0 : tensor<1x13x13x64xi8>
}