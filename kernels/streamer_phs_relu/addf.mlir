#map = affine_map<(m) -> (m)>
func.func @relu(%a: tensor<16xf32>, %b: tensor<16xf32>) -> tensor<16xf32> {
    %out = tensor.empty() : tensor<16xf32>
    %added = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%a, %b : tensor<16xf32>, tensor<16xf32>) outs(%out: tensor<16xf32>) attrs={phs_acc=@acc1} {
    ^bb0(%a_it: f32, %b_it : f32, %out_it: f32):
        %add = arith.addf %a_it, %b_it : f32
        linalg.yield %add: f32
    } -> tensor<16xf32>
    return %added : tensor<16xf32>
}
