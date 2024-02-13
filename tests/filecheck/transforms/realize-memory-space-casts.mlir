// RUN: ./compiler/snax-opt --split-input-file %s -p realize-memory-space-casts --print-op-generic | filecheck %s

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
// CHECK-NEXT:     "memref.copy"(%arg0, %0) : (memref<64xi32, 0 : i32>, memref<64xi32, 1 : i32>) -> ()
// CHECK-NEXT:     "memref.copy"(%arg1, %1) : (memref<64xi32, 0 : i32>, memref<64xi32, 1 : i32>) -> ()
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
// CHECK-NEXT:     "memref.copy"(%arg0, %0) : (memref<64xi32, 0 : i32>, memref<64xi32, 1 : i32>) -> ()
// CHECK-NEXT:     "memref.copy"(%arg1, %1) : (memref<64xi32, 0 : i32>, memref<64xi32, 1 : i32>) -> ()
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
// CHECK-NEXT:     "memref.copy"(%arg0, %2) : (memref<?xi32, 0 : i32>, memref<?xi32, 1 : i32>) -> ()
// CHECK-NEXT:     "memref.copy"(%arg1, %5) : (memref<?xi32, 0 : i32>, memref<?xi32, 1 : i32>) -> ()
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
