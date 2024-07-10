"builtin.module"() ({
  "func.func"() <{function_type = (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32>) -> (), sym_name = "tiled_matmul"}> ({
  ^bb0(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16]>>, %arg2: memref<16x16xi32>):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 0 : index}> : () -> index
    %2 = "arith.constant"() <{value = 1 : index}> : () -> index
    %3 = "arith.constant"() <{value = 16 : index}> : () -> index
    %4 = "arith.constant"() <{value = 8 : index}> : () -> index
    "scf.for"(%1, %3, %4) ({
    ^bb0(%arg3: index):
      "scf.for"(%1, %3, %4) ({
      ^bb0(%arg4: index):
        "scf.for"(%1, %3, %4) ({
        ^bb0(%arg5: index):
          %5 = "memref.subview"(%arg0, %arg3, %arg4, %4, %4, %2, %2) <{operandSegmentSizes = array<i32: 1, 2, 2, 2>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>}> : (memref<16x16xi8>, index, index, index, index, index, index) -> memref<?x?xi8, strided<[?, ?], offset: ?>>
          %6 = "memref.subview"(%arg1, %arg4, %arg5, %4, %4, %2, %2) <{operandSegmentSizes = array<i32: 1, 2, 2, 2>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>}> : (memref<16x16xi8, strided<[1, 16]>>, index, index, index, index, index, index) -> memref<?x?xi8, strided<[?, ?], offset: ?>>
          %7 = "memref.subview"(%arg2, %arg3, %arg5, %4, %4, %2, %2) <{operandSegmentSizes = array<i32: 1, 2, 2, 2>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>}> : (memref<16x16xi32>, index, index, index, index, index, index) -> memref<?x?xi32, strided<[?, ?], offset: ?>>
          "linalg.generic"(%5, %6, %0, %0, %7) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 4, 1>}> ({
          ^bb0(%arg6: i8, %arg7: i8, %arg8: i32, %arg9: i32, %arg10: i32):
            %8 = "arith.extsi"(%arg6) : (i8) -> i32
            %9 = "arith.subi"(%8, %arg8) : (i32, i32) -> i32
            %10 = "arith.extsi"(%arg7) : (i8) -> i32
            %11 = "arith.subi"(%10, %arg9) : (i32, i32) -> i32
            %12 = "arith.muli"(%9, %11) : (i32, i32) -> i32
            %13 = "arith.addi"(%arg10, %12) : (i32, i32) -> i32
            "linalg.yield"(%13) : (i32) -> ()
          }) : (memref<?x?xi8, strided<[?, ?], offset: ?>>, memref<?x?xi8, strided<[?, ?], offset: ?>>, i32, i32, memref<?x?xi32, strided<[?, ?], offset: ?>>) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
