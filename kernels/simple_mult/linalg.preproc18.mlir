"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>):
    "linalg.generic"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %0 = "arith.muli"(%arg3, %arg4) : (i32, i32) -> i32
      "linalg.yield"(%0) : (i32) -> ()
    }) {"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>} : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), sym_name = "simple_mult", sym_visibility = "public"} : () -> ()
}) : () -> ()

