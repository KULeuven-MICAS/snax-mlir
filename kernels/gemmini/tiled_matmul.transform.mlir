#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> ()>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @gemmini_matmul(%arg0: memref<128x128xi8>, %arg1: memref<128x128xi8>, %arg2: memref<128x128xi32>, %arg3: memref<128x128xi32>) {
    %c0_i32 = arith.constant 0 : i32
    linalg.generic {indexing_maps = [#map, #map1, #map2, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], first, library_call="gemmini"} ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<128x128xi8>, memref<128x128xi8>, i32, i32) outs(%arg2 : memref<128x128xi32>) {
    ^bb0(%in: i8, %in_0: i8, %in_1: i32, %in_2: i32, %out: i32):
      %0 = arith.extsi %in : i8 to i32
      %1 = arith.subi %0, %in_1 : i32
      %2 = arith.extsi %in_0 : i8 to i32
      %3 = arith.subi %2, %in_2 : i32
      %4 = arith.muli %1, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    }
    return
  }
  module @transforms attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match attributes {first} in %arg0 : (!transform.any_op) -> !transform.any_op
      %tiled_linalg_op, %loops:3 = transform.structured.tile_using_for %0 [80, 96, 128] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield 
    }
  }
}
