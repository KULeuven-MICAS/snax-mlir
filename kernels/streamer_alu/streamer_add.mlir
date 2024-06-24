#streamer_add_attributes = {
  indexing_maps = [
    affine_map<(n) -> (n)>,
    affine_map<(n) -> (n)>,
    affine_map<(n) -> (n)>
  ],
  iterator_types = ["parallel"],
  library_call = "snax_alu"
}

func.func public @streamer_add(%A: memref<?xi64>,
                             %B: memref<?xi64>,
                             %D: memref<?xi64>) -> () {
  linalg.generic #streamer_add_attributes
  ins(%A, %B: memref<?xi64>, memref<?xi64>)
  outs(%D: memref<?xi64>) {
  ^bb0(%a: i64, %b: i64, %d: i64):
    %r0 = arith.addi %a, %b : i64
    linalg.yield %r0 : i64
  }
  return
}
