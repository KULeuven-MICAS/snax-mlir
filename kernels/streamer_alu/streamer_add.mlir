module {
  func.func public @streamer_add(%arg0: memref<?xi64>, %arg1: memref<?xi64>, %arg2: memref<?xi64>) {
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"], library_call = "snax_alu"} ins(%arg0, %arg1 : memref<?xi64>, memref<?xi64>) outs(%arg2 : memref<?xi64>) {
    ^bb0(%in: i64, %in_0: i64, %out: i64):
      %0 = arith.addi %in, %in_0 : i64
      linalg.yield %0 : i64
    }
    return
  }
}
