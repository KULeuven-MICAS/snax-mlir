builtin.module {
  func.func public @rescale_down(%arg0 : tensor<1x114x114x64xi8>) -> tensor<1x56x56x64xi8> {
    %c-128_i8 = arith.constant -128 : i8
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<1x114x114x64xi8>
    %0 = tensor.empty(%dim) : tensor<1x56x56x64xi8>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%c-128_i8 : i8) outs(%0 : tensor<1x56x56x64xi8>) {
    ^0(%in : i8, %out : i8):
      linalg.yield %in : i8
    } -> tensor<1x56x56x64xi8>
    %2 = tensor.empty() : tensor<3x3xi8>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, ((d1 * 2) + d4), ((d2 * 2) + d5), d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %2 : tensor<1x114x114x64xi8>, tensor<3x3xi8>) outs(%1 : tensor<1x56x56x64xi8>) {
    ^1(%in_1 : i8, %in_2 : i8, %out_1 : i8):
      %4 = arith.maxsi %out_1, %in_1 : i8
      linalg.yield %4 : i8
    } -> tensor<1x56x56x64xi8>
    func.return %3 : tensor<1x56x56x64xi8>
  }
}
