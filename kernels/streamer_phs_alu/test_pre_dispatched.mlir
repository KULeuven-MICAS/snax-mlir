#map = affine_map<(d0) -> (d0)>
module {
  func.func public @streamer_add(%arg0: memref<16xi64>, %arg1: memref<16xi64>, %arg2: memref<16xi64>) {
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"], library_call = "acc1_stream"} ins(%arg0, %arg1 : memref<16xi64>, memref<16xi64>) outs(%arg2 : memref<16xi64>) {
    ^bb0(%in: i64, %in_0: i64, %out: i64):
      %0 = arith.addi %in, %in_0 : i64
      linalg.yield %0 : i64
    }
    return
  }
}
