#simple_mult_attributes = {
  indexing_maps = [
    affine_map<(n) -> (n)>,
    affine_map<(n) -> (n)>,
    affine_map<(n) -> (n)>
  ],
  iterator_types = ["parallel"]
}

// https://github.com/pulp-platform/hwpe-mac-engine
// in 'simple_mult' mode, it takes two 32bit fixed-point streams (vectors),
// A, B and computes D = A * B where '*' is the elementwise product.
func.func public @simple_mult(%A: memref<?xi32>,
                             %B: memref<?xi32>,
                             %D: memref<?xi32>) -> () {
  linalg.generic #simple_mult_attributes
  ins(%A, %B: memref<?xi32>, memref<?xi32>)
  outs(%D: memref<?xi32>) {
  ^bb0(%a: i32, %b: i32, %d: i32):
    %r0 = arith.muli %a, %b : i32
    linalg.yield %r0 : i32
  }
  return
}
