#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d4, d2 + d5, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map9 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map10 = affine_map<(d0, d1, d2) -> ()>
#map11 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @toy(%arg0 : tensor<1x18x18x16xi8>, %arg1 : tensor<16x3x3x16xi8>, %arg2 : tensor<16x10xi8>) -> tensor<1x10xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %return = tensor.empty() : tensor<1x10xi32>
  %return_for = scf.for %i = %c0 to %c10 step %c1 iter_args(%return_for = %return) -> tensor<1x10xi32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<1x16x16x16xi32>
    %conv = linalg.conv_2d_nhwc_fhwc_q ins(%arg0, %arg1, %c0_i32, %c0_i32 : tensor<1x18x18x16xi8>, tensor<16x3x3x16xi8>, i32, i32) outs(%0 : tensor<1x16x16x16xi32>) -> tensor<1x16x16x16xi32>
    %1 = tensor.empty() : tensor<1x16x16x16xi8>
    %conv_q = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv : tensor<1x16x16x16xi32>) outs(%1 : tensor<1x16x16x16xi8>) {
    ^bb0(%in: i32, %out: i8):
      %a = "kernel.rescale"(%in) {double_round = true, input_zp = 23 : i8, max_int = 100 : i8, min_int = -110 : i8, multiplier = 1234567890 : i32, output_zp = -15 : i8, shift = 39 : i8} : (i32) -> i8
      linalg.yield %a : i8
    } -> tensor<1x16x16x16xi8>
    %2 = tensor.empty() : tensor<1x1x1x16xi8>
    %not_used = tensor.empty() : tensor<16x16xi8>
    %pooled = linalg.generic {indexing_maps = [#map2, #map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%conv_q, %not_used : tensor<1x16x16x16xi8>, tensor<16x16xi8>) outs(%2 : tensor<1x1x1x16xi8>) {
      ^bb0(%in: i8, %in2: i8, %out: i8):
        %b = arith.maxsi %out, %in : i8
        linalg.yield %b : i8
      } -> tensor<1x1x1x16xi8>
    %collapsed = tensor.collapse_shape %pooled [[0, 1, 2], [3]] : tensor<1x1x1x16xi8> into tensor<1x16xi8>
    %3 = tensor.empty() : tensor<1x10xi32>
    %result = linalg.generic {indexing_maps = [#map8, #map9, #map10, #map10, #map11], iterator_types = ["parallel", "parallel", "reduction"], library_call="none"} ins(%collapsed, %arg2, %c0_i32, %c0_i32 : tensor<1x16xi8>, tensor<16x10xi8>, i32, i32) outs(%3 : tensor<1x10xi32>) {
    ^bb0(%in: i8, %in_0: i8, %in_1: i32, %in_2: i32, %out: i32):
      %9 = arith.extsi %in : i8 to i32
      %10 = arith.subi %9, %in_1 : i32
      %11 = arith.extsi %in_0 : i8 to i32
      %12 = arith.subi %11, %in_2 : i32
      %13 = arith.muli %10, %12 : i32
      %14 = arith.addi %out, %13 : i32
      linalg.yield %14 : i32
    } -> tensor<1x10xi32>
    scf.yield %result: tensor<1x10xi32>
  }
func.return %return_for : tensor<1x10xi32>
}
