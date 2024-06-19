#simple_mult_attributes = {
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
                             %D: memref<?xi128>) -> () {
  linalg.generic #simple_mult_attributes
  ins(%A, %B: memref<?xi64>, memref<?xi64>)
  outs(%D: memref<?xi128>) {
  ^bb0(%a: i64, %b: i64, %d: i128):
    %a_ext = arith.extsi %a: i64 to i128
    %b_ext = arith.extsi %b: i64 to i128
    %r0 = arith.addi %a_ext, %b_ext : i128
    linalg.yield %r0 : i128
  }
  return
}
