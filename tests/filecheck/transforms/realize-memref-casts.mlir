// RUN: ./compiler/snax-opt --split-input-file %s -p realize-memref-casts --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "test.op"() : () -> (memref<64xi32, 0 : i32>)
  %1 = "memref.memory_space_cast"(%0) : (memref<64xi32, 0 : i32>) -> memref<64xi32, 1 : i32>
  "test.op"(%1) : (memref<64xi32, 1 : i32>) -> ()
}): () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "test.op"() : () -> memref<64xi32, 0 : i32>
// CHECK-NEXT:   %1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64xi32, 1 : i32>
// CHECK-NEXT:   "memref.copy"(%0, %1) : (memref<64xi32, 0 : i32>, memref<64xi32, 1 : i32>) -> ()
// CHECK-NEXT:   "test.op"(%1) : (memref<64xi32, 1 : i32>) -> ()
// CHECK-NEXT:   "memref.copy"(%1, %0) : (memref<64xi32, 1 : i32>, memref<64xi32, 0 : i32>) -> ()
// CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  %0 = "test.op"() : () -> (memref<64xi32, 1 : i32>)
  %1 = "snax.layout_cast"(%0) : (memref<64xi32, 1 : i32>) -> memref<64xi32, strided<[1, 8]>, 1 : i32>
  "test.op"(%1) : (memref<64xi32, strided<[1, 8]>, 1 : i32>) -> ()
}): () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "test.op"() : () -> memref<64xi32, 1 : i32>
// CHECK-NEXT:   %1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64xi32, strided<[1, 8]>, 1 : i32>
// CHECK-NEXT:   "memref.copy"(%0, %1) : (memref<64xi32, 1 : i32>, memref<64xi32, strided<[1, 8]>, 1 : i32>) -> ()
// CHECK-NEXT:   "test.op"(%1) : (memref<64xi32, strided<[1, 8]>, 1 : i32>) -> ()
// CHECK-NEXT:   "memref.copy"(%1, %0) : (memref<64xi32, strided<[1, 8]>, 1 : i32>, memref<64xi32, 1 : i32>) -> ()
// CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  %0 = "test.op"() : () -> (memref<64xi32, 0 : i32>)
  %1 = "memref.memory_space_cast"(%0) : (memref<64xi32, 0 : i32>) -> memref<64xi32, 1 : i32>
  %2 = "snax.layout_cast"(%1) : (memref<64xi32, 1 : i32>) -> memref<64xi32, strided<[1, 8]>, 1 : i32>
  "test.op"(%2) : (memref<64xi32, strided<[1, 8]>, 1 : i32>) -> ()
}): () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "test.op"() : () -> memref<64xi32, 0 : i32>
// CHECK-NEXT:   %1 = "memref.memory_space_cast"(%0) : (memref<64xi32, 0 : i32>) -> memref<64xi32, 1 : i32>
// CHECK-NEXT:   %2 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64xi32, strided<[1, 8]>, 1 : i32>
// CHECK-NEXT:   "memref.copy"(%0, %2) : (memref<64xi32, 0 : i32>, memref<64xi32, strided<[1, 8]>, 1 : i32>) -> ()
// CHECK-NEXT:   "test.op"(%2) : (memref<64xi32, strided<[1, 8]>, 1 : i32>) -> ()
// CHECK-NEXT:   "memref.copy"(%2, %0) : (memref<64xi32, strided<[1, 8]>, 1 : i32>, memref<64xi32, 0 : i32>) -> ()
// CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32, 0 : i32>, memref<64xi32, 0 : i32>, memref<64xi32, 0 : i32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<64xi32, 0 : i32>, %arg1 : memref<64xi32, 0 : i32>, %arg2 : memref<64xi32, 0 : i32>):
    %0 = "memref.memory_space_cast"(%arg0) : (memref<64xi32, 0 : i32>) -> memref<64xi32, 1 : i32>
    %1 = "memref.memory_space_cast"(%arg1) : (memref<64xi32, 0 : i32>) -> memref<64xi32, 1 : i32>
    %2 = "memref.memory_space_cast"(%arg2) : (memref<64xi32, 0 : i32>) -> memref<64xi32, 1 : i32>
    "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
    ^1(%arg3 : i32, %arg4 : i32, %arg5 : i32):
      %3 = "arith.muli"(%arg3, %arg4) : (i32, i32) -> i32
      "linalg.yield"(%3) : (i32) -> ()
    }) : (memref<64xi32, 1 : i32>, memref<64xi32, 1 : i32>, memref<64xi32, 1 : i32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32, 0 : i32>, memref<64xi32, 0 : i32>, memref<64xi32, 0 : i32>) -> (), "sym_visibility" = "public"}> ({
// CHECK-NEXT:   ^0(%arg0 : memref<64xi32, 0 : i32>, %arg1 : memref<64xi32, 0 : i32>, %arg2 : memref<64xi32, 0 : i32>):
// CHECK-NEXT:     %0 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64xi32, 1 : i32>
// CHECK-NEXT:     %1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64xi32, 1 : i32>
// CHECK-NEXT:     %2 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64xi32, 1 : i32>
// CHECK-NEXT:     "memref.copy"(%arg1, %1) : (memref<64xi32, 0 : i32>, memref<64xi32, 1 : i32>) -> ()
// CHECK-NEXT:     "memref.copy"(%arg0, %0) : (memref<64xi32, 0 : i32>, memref<64xi32, 1 : i32>) -> ()
// CHECK-NEXT:     "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
// CHECK-NEXT:     ^1(%arg3 : i32, %arg4 : i32, %arg5 : i32):
// CHECK-NEXT:       %3 = "arith.muli"(%arg3, %arg4) : (i32, i32) -> i32
// CHECK-NEXT:       "linalg.yield"(%3) : (i32) -> ()
// CHECK-NEXT:     }) : (memref<64xi32, 1 : i32>, memref<64xi32, 1 : i32>, memref<64xi32, 1 : i32>) -> ()
// CHECK-NEXT:     "memref.copy"(%2, %arg2) : (memref<64xi32, 1 : i32>, memref<64xi32, 0 : i32>) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32, 0 : i32>, memref<64xi32, 0 : i32>, memref<64xi32, 0 : i32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<64xi32, 0 : i32>, %arg1 : memref<64xi32, 0 : i32>, %arg2 : memref<64xi32, 0 : i32>):
    %0 = "memref.memory_space_cast"(%arg0) : (memref<64xi32, 0 : i32>) -> memref<64xi32, 1 : i32>
    %1 = "memref.memory_space_cast"(%arg1) : (memref<64xi32, 0 : i32>) -> memref<64xi32, 1 : i32>
    %2 = "memref.memory_space_cast"(%arg2) : (memref<64xi32, 0 : i32>) -> memref<64xi32, 1 : i32>
    "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
    ^1(%arg3 : i32, %arg4 : i32, %arg5 : i32):
      %3 = "arith.muli"(%arg3, %arg4) : (i32, i32) -> i32
      "linalg.yield"(%3) : (i32) -> ()
    }) : (memref<64xi32, 1 : i32>, memref<64xi32, 1 : i32>, memref<64xi32, 1 : i32>) -> ()
    "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
    ^2(%arg3_1 : i32, %arg4_1 : i32, %arg5_1 : i32):
      %4 = "arith.muli"(%arg3_1, %arg4_1) : (i32, i32) -> i32
      "linalg.yield"(%4) : (i32) -> ()
    }) : (memref<64xi32, 1 : i32>, memref<64xi32, 1 : i32>, memref<64xi32, 1 : i32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32, 0 : i32>, memref<64xi32, 0 : i32>, memref<64xi32, 0 : i32>) -> (), "sym_visibility" = "public"}> ({
// CHECK-NEXT:   ^0(%arg0 : memref<64xi32, 0 : i32>, %arg1 : memref<64xi32, 0 : i32>, %arg2 : memref<64xi32, 0 : i32>):
// CHECK-NEXT:     %0 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64xi32, 1 : i32>
// CHECK-NEXT:     %1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64xi32, 1 : i32>
// CHECK-NEXT:     %2 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64xi32, 1 : i32>
// CHECK-NEXT:     "memref.copy"(%arg1, %1) : (memref<64xi32, 0 : i32>, memref<64xi32, 1 : i32>) -> ()
// CHECK-NEXT:     "memref.copy"(%arg0, %0) : (memref<64xi32, 0 : i32>, memref<64xi32, 1 : i32>) -> ()
// CHECK-NEXT:     "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
// CHECK-NEXT:     ^1(%arg3 : i32, %arg4 : i32, %arg5 : i32):
// CHECK-NEXT:       %3 = "arith.muli"(%arg3, %arg4) : (i32, i32) -> i32
// CHECK-NEXT:       "linalg.yield"(%3) : (i32) -> ()
// CHECK-NEXT:     }) : (memref<64xi32, 1 : i32>, memref<64xi32, 1 : i32>, memref<64xi32, 1 : i32>) -> ()
// CHECK-NEXT:     "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
// CHECK-NEXT:     ^2(%arg3_1 : i32, %arg4_1 : i32, %arg5_1 : i32):
// CHECK-NEXT:       %4 = "arith.muli"(%arg3_1, %arg4_1) : (i32, i32) -> i32
// CHECK-NEXT:       "linalg.yield"(%4) : (i32) -> ()
// CHECK-NEXT:     }) : (memref<64xi32, 1 : i32>, memref<64xi32, 1 : i32>, memref<64xi32, 1 : i32>) -> ()
// CHECK-NEXT:     "memref.copy"(%2, %arg2) : (memref<64xi32, 1 : i32>, memref<64xi32, 0 : i32>) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<?xi32, 0 : i32>, memref<?xi32, 0 : i32>, memref<?xi32, 0 : i32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<?xi32, 0 : i32>, %arg1 : memref<?xi32, 0 : i32>, %arg2 : memref<?xi32, 0 : i32>):
    %0 = "memref.memory_space_cast"(%arg0) : (memref<?xi32, 0 : i32>) -> memref<?xi32, 1 : i32>
    %1 = "memref.memory_space_cast"(%arg1) : (memref<?xi32, 0 : i32>) -> memref<?xi32, 1 : i32>
    %2 = "memref.memory_space_cast"(%arg2) : (memref<?xi32, 0 : i32>) -> memref<?xi32, 1 : i32>
    "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
    ^1(%arg3 : i32, %arg4 : i32, %arg5 : i32):
      %3 = "arith.muli"(%arg3, %arg4) : (i32, i32) -> i32
      "linalg.yield"(%3) : (i32) -> ()
    }) : (memref<?xi32, 1 : i32>, memref<?xi32, 1 : i32>, memref<?xi32, 1 : i32>) -> ()
    "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
    ^2(%arg3_1 : i32, %arg4_1 : i32, %arg5_1 : i32):
      %4 = "arith.muli"(%arg3_1, %arg4_1) : (i32, i32) -> i32
      "linalg.yield"(%4) : (i32) -> ()
    }) : (memref<?xi32, 1 : i32>, memref<?xi32, 1 : i32>, memref<?xi32, 1 : i32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<?xi32, 0 : i32>, memref<?xi32, 0 : i32>, memref<?xi32, 0 : i32>) -> (), "sym_visibility" = "public"}> ({
// CHECK-NEXT:   ^0(%arg0 : memref<?xi32, 0 : i32>, %arg1 : memref<?xi32, 0 : i32>, %arg2 : memref<?xi32, 0 : i32>):
// CHECK-NEXT:     %0 = "arith.constant"() <{"value" = 0 : index}> : () -> index
// CHECK-NEXT:     %1 = "memref.dim"(%arg0, %0) : (memref<?xi32, 0 : i32>, index) -> index
// CHECK-NEXT:     %2 = "memref.alloc"(%1) <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 1, 0>}> : (index) -> memref<?xi32, 1 : i32>
// CHECK-NEXT:     %3 = "arith.constant"() <{"value" = 0 : index}> : () -> index
// CHECK-NEXT:     %4 = "memref.dim"(%arg1, %3) : (memref<?xi32, 0 : i32>, index) -> index
// CHECK-NEXT:     %5 = "memref.alloc"(%4) <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 1, 0>}> : (index) -> memref<?xi32, 1 : i32>
// CHECK-NEXT:     %6 = "arith.constant"() <{"value" = 0 : index}> : () -> index
// CHECK-NEXT:     %7 = "memref.dim"(%arg2, %6) : (memref<?xi32, 0 : i32>, index) -> index
// CHECK-NEXT:     %8 = "memref.alloc"(%7) <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 1, 0>}> : (index) -> memref<?xi32, 1 : i32>
// CHECK-NEXT:     "memref.copy"(%arg1, %5) : (memref<?xi32, 0 : i32>, memref<?xi32, 1 : i32>) -> ()
// CHECK-NEXT:     "memref.copy"(%arg0, %2) : (memref<?xi32, 0 : i32>, memref<?xi32, 1 : i32>) -> ()
// CHECK-NEXT:     "linalg.generic"(%2, %5, %8) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
// CHECK-NEXT:     ^1(%arg3 : i32, %arg4 : i32, %arg5 : i32):
// CHECK-NEXT:       %9 = "arith.muli"(%arg3, %arg4) : (i32, i32) -> i32
// CHECK-NEXT:       "linalg.yield"(%9) : (i32) -> ()
// CHECK-NEXT:     }) : (memref<?xi32, 1 : i32>, memref<?xi32, 1 : i32>, memref<?xi32, 1 : i32>) -> ()
// CHECK-NEXT:     "linalg.generic"(%2, %5, %8) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
// CHECK-NEXT:     ^2(%arg3_1 : i32, %arg4_1 : i32, %arg5_1 : i32):
// CHECK-NEXT:       %10 = "arith.muli"(%arg3_1, %arg4_1) : (i32, i32) -> i32
// CHECK-NEXT:       "linalg.yield"(%10) : (i32) -> ()
// CHECK-NEXT:     }) : (memref<?xi32, 1 : i32>, memref<?xi32, 1 : i32>, memref<?xi32, 1 : i32>) -> ()
// CHECK-NEXT:     "memref.copy"(%8, %arg2) : (memref<?xi32, 1 : i32>, memref<?xi32, 0 : i32>) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  "func.func"() <{"sym_name" = "mnist", "function_type" = (memref<?x128xi8, 0 : i32>, memref<128x128xi8, 0 : i32>, memref<?x128xi32, 0 : i32>) -> memref<?x128xi32, 0 : i32>}> ({
  ^0(%arg0 : memref<?x128xi8, 0 : i32>, %arg1 : memref<128x128xi8, 0 : i32>, %arg2 : memref<?x128xi32, 0 : i32>):
    %0 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
    %1 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
    %2 = "memref.memory_space_cast"(%arg0) : (memref<?x128xi8, 0 : i32>) -> memref<?x128xi8, 1 : i32>
    %3 = "memref.memory_space_cast"(%arg1) : (memref<128x128xi8, 0 : i32>) -> memref<128x128xi8, 1 : i32>
    %4 = "memref.memory_space_cast"(%arg2) : (memref<?x128xi32, 0 : i32>) -> memref<?x128xi32, 1 : i32>
    %5 = "snax.layout_cast"(%2) : (memref<?x128xi8, 1 : i32>) -> memref<?x128xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, 1 : i32>
    %6 = "snax.layout_cast"(%3) : (memref<128x128xi8, 1 : i32>) -> memref<128x128xi8, #tsl.tsl<[?, 8] -> (256, 1), [?, 8] -> (?, 8), offset: 64>, 1 : i32>
    %7 = "snax.layout_cast"(%4) : (memref<?x128xi32, 1 : i32>) -> memref<?x128xi32, #tsl.tsl<[?, 8] -> (256, 4), [?, 8] -> (?, 32)>, 1 : i32>
    "linalg.generic"(%5, %6, %0, %1, %7) <{"indexing_maps" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], "iterator_types" = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], "operandSegmentSizes" = array<i32: 4, 1>}> ({
    ^1(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
      %8 = "arith.extsi"(%arg3) : (i8) -> i32
      %9 = "arith.subi"(%8, %arg5) : (i32, i32) -> i32
      %10 = "arith.extsi"(%arg4) : (i8) -> i32
      %11 = "arith.subi"(%10, %arg6) : (i32, i32) -> i32
      %12 = "arith.muli"(%9, %11) : (i32, i32) -> i32
      %13 = "arith.addi"(%arg7, %12) : (i32, i32) -> i32
      "linalg.yield"(%13) : (i32) -> ()
    }) {"library_call" = "snax_qgemm"} : (memref<?x128xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, 1 : i32>, memref<128x128xi8, #tsl.tsl<[?, 8] -> (256, 1), [?, 8] -> (?, 8), offset: 64>, 1 : i32>, i32, i32, memref<?x128xi32, #tsl.tsl<[?, 8] -> (256, 4), [?, 8] -> (?, 32)>, 1 : i32>) -> ()
    "func.return"(%arg2) : (memref<?x128xi32, 0 : i32>) -> ()
  }) : () -> ()
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:  "func.func"() <{"sym_name" = "mnist", "function_type" = (memref<?x128xi8, 0 : i32>, memref<128x128xi8, 0 : i32>, memref<?x128xi32, 0 : i32>) -> memref<?x128xi32, 0 : i32>}> ({
// CHECK-NEXT:  ^0(%arg0 : memref<?x128xi8, 0 : i32>, %arg1 : memref<128x128xi8, 0 : i32>, %arg2 : memref<?x128xi32, 0 : i32>):
// CHECK-NEXT:    %0 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
// CHECK-NEXT:    %1 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
// CHECK-NEXT:    %2 = "memref.memory_space_cast"(%arg0) : (memref<?x128xi8, 0 : i32>) -> memref<?x128xi8, 1 : i32>
// CHECK-NEXT:    %3 = "memref.memory_space_cast"(%arg1) : (memref<128x128xi8, 0 : i32>) -> memref<128x128xi8, 1 : i32>
// CHECK-NEXT:    %4 = "memref.memory_space_cast"(%arg2) : (memref<?x128xi32, 0 : i32>) -> memref<?x128xi32, 1 : i32>
// CHECK-NEXT:    %5 = "arith.constant"() <{"value" = 0 : index}> : () -> index
// CHECK-NEXT:    %6 = "memref.dim"(%arg0, %5) : (memref<?x128xi8, 0 : i32>, index) -> index
// CHECK-NEXT:    %7 = "memref.alloc"(%6) <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 1, 0>}> : (index) -> memref<?x128xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, 1 : i32>
// CHECK-NEXT:    %8 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<128x128xi8, #tsl.tsl<[?, 8] -> (256, 1), [?, 8] -> (?, 8), offset: 64>, 1 : i32>
// CHECK-NEXT:    %9 = "arith.constant"() <{"value" = 0 : index}> : () -> index
// CHECK-NEXT:    %10 = "memref.dim"(%arg2, %9) : (memref<?x128xi32, 0 : i32>, index) -> index
// CHECK-NEXT:    %11 = "memref.alloc"(%10) <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 1, 0>}> : (index) -> memref<?x128xi32, #tsl.tsl<[?, 8] -> (256, 4), [?, 8] -> (?, 32)>, 1 : i32>
// CHECK-NEXT:    "memref.copy"(%arg1, %8) : (memref<128x128xi8, 0 : i32>, memref<128x128xi8, #tsl.tsl<[?, 8] -> (256, 1), [?, 8] -> (?, 8), offset: 64>, 1 : i32>) -> ()
// CHECK-NEXT:    "memref.copy"(%arg0, %7) : (memref<?x128xi8, 0 : i32>, memref<?x128xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, 1 : i32>) -> ()
// CHECK-NEXT:    "linalg.generic"(%7, %8, %0, %1, %11) <{"indexing_maps" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], "iterator_types" = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], "operandSegmentSizes" = array<i32: 4, 1>}> ({
// CHECK-NEXT:    ^1(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
// CHECK-NEXT:      %12 = "arith.extsi"(%arg3) : (i8) -> i32
// CHECK-NEXT:      %13 = "arith.subi"(%12, %arg5) : (i32, i32) -> i32
// CHECK-NEXT:      %14 = "arith.extsi"(%arg4) : (i8) -> i32
// CHECK-NEXT:      %15 = "arith.subi"(%14, %arg6) : (i32, i32) -> i32
// CHECK-NEXT:      %16 = "arith.muli"(%13, %15) : (i32, i32) -> i32
// CHECK-NEXT:      %17 = "arith.addi"(%arg7, %16) : (i32, i32) -> i32
// CHECK-NEXT:      "linalg.yield"(%17) : (i32) -> ()
// CHECK-NEXT:    }) {"library_call" = "snax_qgemm"} : (memref<?x128xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, 1 : i32>, memref<128x128xi8, #tsl.tsl<[?, 8] -> (256, 1), [?, 8] -> (?, 8), offset: 64>, 1 : i32>, i32, i32, memref<?x128xi32, #tsl.tsl<[?, 8] -> (256, 4), [?, 8] -> (?, 32)>, 1 : i32>) -> ()
// CHECK-NEXT:    "memref.copy"(%11, %arg2) : (memref<?x128xi32, #tsl.tsl<[?, 8] -> (256, 4), [?, 8] -> (?, 32)>, 1 : i32>, memref<?x128xi32, 0 : i32>) -> ()
// CHECK-NEXT:    "func.return"(%arg2) : (memref<?x128xi32, 0 : i32>) -> ()
// CHECK-NEXT:  }) : () -> ()
// CHECK-NEXT:}) : () -> ()


