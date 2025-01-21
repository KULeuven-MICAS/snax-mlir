"builtin.module"() ({
  "func.func"() <{function_type = (memref<16x16xi32>, memref<16x16xi8>) -> (), sym_name = "rescale", sym_visibility = "public"}> ({
  ^bb0(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi8>):
    "linalg.generic"(%arg0, %arg1) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg2: i32, %arg3: i8):
      %0 = "kernel.rescale"(%arg2) {double_round = true, input_zp = 23 : i8, max_int = 100 : i8, min_int = -110 : i8, multiplier = 1234567890 : i32, output_zp = -15 : i8, shift = 39 : i8} : (i32) -> i8
      "linalg.yield"(%0) : (i8) -> ()
    }) : (memref<16x16xi32>, memref<16x16xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
