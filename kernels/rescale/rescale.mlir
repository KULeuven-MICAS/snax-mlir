"builtin.module"() ({
  "func.func"() <{function_type = (memref<16x16xi32>, memref<16x16xi8>) -> (), sym_name = "rescale", sym_visibility = "public"}> ({
  ^bb0(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi8>):
    %c_256 = arith.constant 256 : index
    %c_16 = arith.constant 16 : index
    %zero_row = "snax.alloc"(%c_256, %c_16, %c_16) <{memory_space = "L1", alignment = 64 : i64}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %zero_memref = builtin.unrealized_conversion_cast %zero_row : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)> to memref<16x16xi32>
    "linalg.generic"(%arg0, %arg1) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg2: i32, %arg3: i8):
      %0 = "kernel.rescale"(%arg2) {double_round = true, input_zp = 23 : i8, max_int = 100 : i8, min_int = -110 : i8, multiplier = 1234567890 : i32, output_zp = -15 : i8, shift = 39 : i8} : (i32) -> i8
      "linalg.yield"(%0) : (i8) -> ()
    }) : (memref<16x16xi32>, memref<16x16xi8>) -> ()
    "memref.copy" (%zero_memref, %zero_memref) : (memref<16x16xi32>, memref<16x16xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
