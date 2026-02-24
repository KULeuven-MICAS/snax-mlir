#map = affine_map<(d0) -> (d0)>
module {
  func.func public @streamer_add(%arg0: tensor<16xf64>, %arg1: tensor<16xf64>) -> tensor<16xf64> {
    %0 = tensor.empty() : tensor<16xf64>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<16xf64>, tensor<16xf64>) outs(%0 : tensor<16xf64>) attrs={phs_acc=@acc1}{
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %6 = arith.addf %in, %in_0 : f64
      linalg.yield %6 : f64
    } -> tensor<16xf64>
    %2 = tensor.empty() : tensor<16xf64>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %1 : tensor<16xf64>, tensor<16xf64>) outs(%2 : tensor<16xf64>) attrs={phs_acc=@acc1}{
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %6 = arith.subf %in, %in_0 : f64
      linalg.yield %6 : f64
    } -> tensor<16xf64>
    %4 = tensor.empty() : tensor<16xf64>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %3 : tensor<16xf64>, tensor<16xf64>) outs(%4 : tensor<16xf64>) attrs={phs_acc=@acc1}{
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %6 = arith.mulf %in, %in_0 : f64
      linalg.yield %6 : f64
    } -> tensor<16xf64>
    return %5 : tensor<16xf64>
  }
}
