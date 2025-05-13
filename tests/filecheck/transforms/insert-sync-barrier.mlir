// RUN: snax-opt --split-input-file %s -p insert-sync-barrier --print-op-generic | filecheck %s

// two global ops: no synchronization barrier required
"builtin.module"() ({
  %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> {alignment = 64 : i64} : () -> memref<64xi32>
  "test.op"(%0) : (memref<64xi32>) -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>, alignment = 64 : i64}> : () -> memref<64xi32>
//CHECK-NEXT:   "test.op"(%0) : (memref<64xi32>) -> ()
//CHECK-NEXT: }) : () -> ()

// -----

// one global op, one op dispatched to the dm core, but result not used: no synchronization barrier requried
"builtin.module"() ({
  %0 = "test.op"() : () -> (memref<64xi32>)
  "memref.copy"(%0, %0) : (memref<64xi32>, memref<64xi32>) -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   %0 = "test.op"() : () -> memref<64xi32>
//CHECK-NEXT:   "memref.copy"(%0, %0) : (memref<64xi32>, memref<64xi32>) -> ()
//CHECK-NEXT: }) : () -> ()

// -----

// one global op, one op dispatched to the dm core, result used in global op: synchronization barrier required
"builtin.module"() ({
  %0 = "test.op"() : () -> (memref<64xi32>)
  "memref.copy"(%0, %0) : (memref<64xi32>, memref<64xi32>) -> ()
  "test.op" (%0) : (memref<64xi32>) -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   %0 = "test.op"() : () -> memref<64xi32>
//CHECK-NEXT:   "memref.copy"(%0, %0) : (memref<64xi32>, memref<64xi32>) -> ()
//CHECK-NEXT:   "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:   "test.op"(%0) : (memref<64xi32>) -> ()
//CHECK-NEXT: }) : () -> ()

// -----

// some global ops followed by memcopy dispatched to dm, followed by linalg dispatched to compute, followed by copy dipsatched to dm:
// synchronization barriers required around the linalg
"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32, "L3">, memref<64xi32, "L3">, memref<64xi32, "L3">) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<64xi32, "L3">, %arg1 : memref<64xi32, "L3">, %arg2 : memref<64xi32, "L3">):
    %0 = "memref.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0>}> {"alignment" = 64 : i64} : () -> memref<64xi32, "L1">
    %1 = "memref.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0>}> {"alignment" = 64 : i64} : () -> memref<64xi32, "L1">
    %2 = "memref.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0>}> {"alignment" = 64 : i64} : () -> memref<64xi32, "L1">
    "memref.copy"(%arg0, %0) : (memref<64xi32, "L3">, memref<64xi32, "L1">) -> ()
    "memref.copy"(%arg1, %1) : (memref<64xi32, "L3">, memref<64xi32, "L1">) -> ()
    "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
    ^1(%arg3 : i32, %arg4 : i32, %arg5 : i32):
      %3 = "arith.muli"(%arg3, %arg4) : (i32, i32) -> i32
      "linalg.yield"(%3) : (i32) -> ()
    }) : (memref<64xi32, "L1">, memref<64xi32, "L1">, memref<64xi32, "L1">) -> ()
    "memref.copy"(%2, %arg2) : (memref<64xi32, "L1">, memref<64xi32, "L3">) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   "func.func"() <{sym_name = "simple_mult", function_type = (memref<64xi32, "L3">, memref<64xi32, "L3">, memref<64xi32, "L3">) -> (), sym_visibility = "public"}> ({
//CHECK-NEXT:   ^0(%arg0 : memref<64xi32, "L3">, %arg1 : memref<64xi32, "L3">, %arg2 : memref<64xi32, "L3">):
//CHECK-NEXT:     %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>, alignment = 64 : i64}> : () -> memref<64xi32, "L1">
//CHECK-NEXT:     %1 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>, alignment = 64 : i64}> : () -> memref<64xi32, "L1">
//CHECK-NEXT:     %2 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>, alignment = 64 : i64}> : () -> memref<64xi32, "L1">
//CHECK-NEXT:     "memref.copy"(%arg0, %0) : (memref<64xi32, "L3">, memref<64xi32, "L1">) -> ()
//CHECK-NEXT:     "memref.copy"(%arg1, %1) : (memref<64xi32, "L3">, memref<64xi32, "L1">) -> ()
//CHECK-NEXT:     "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:     "linalg.generic"(%0, %1, %2) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
//CHECK-NEXT:     ^1(%arg3 : i32, %arg4 : i32, %arg5 : i32):
//CHECK-NEXT:       %3 = "arith.muli"(%arg3, %arg4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
//CHECK-NEXT:       "linalg.yield"(%3) : (i32) -> ()
//CHECK-NEXT:     }) : (memref<64xi32, "L1">, memref<64xi32, "L1">, memref<64xi32, "L1">) -> ()
//CHECK-NEXT:     "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:     "memref.copy"(%2, %arg2) : (memref<64xi32, "L1">, memref<64xi32, "L3">) -> ()
//CHECK-NEXT:     "func.return"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()

// -----

// some global ops followed by memcopy dispatched to dm, followed by 2 * linalg dispatched to compute, followed by copy dipsatched to dm:
// synchronization barriers required around the linalgs, but not between as they dispatch on the same core
"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32, "L3">, memref<64xi32, "L3">, memref<64xi32, "L3">) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<64xi32, "L3">, %arg1 : memref<64xi32, "L3">, %arg2 : memref<64xi32, "L3">):
    %0 = "memref.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0>}> {"alignment" = 64 : i64} : () -> memref<64xi32, "L1">
    %1 = "memref.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0>}> {"alignment" = 64 : i64} : () -> memref<64xi32, "L1">
    %2 = "memref.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0>}> {"alignment" = 64 : i64} : () -> memref<64xi32, "L1">
    %3 = "memref.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0>}> {"alignment" = 64 : i64} : () -> memref<64xi32, "L1">
    "memref.copy"(%arg0, %0) : (memref<64xi32, "L3">, memref<64xi32, "L1">) -> ()
    "memref.copy"(%arg1, %1) : (memref<64xi32, "L3">, memref<64xi32, "L1">) -> ()
    "memref.copy"(%arg1, %3) : (memref<64xi32, "L3">, memref<64xi32, "L1">) -> ()
    "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
    ^1(%arg3 : i32, %arg4 : i32, %arg5 : i32):
      %4 = "arith.muli"(%arg3, %arg4) : (i32, i32) -> i32
      "linalg.yield"(%4) : (i32) -> ()
    }) : (memref<64xi32, "L1">, memref<64xi32, "L1">, memref<64xi32, "L1">) -> ()
    "linalg.generic"(%3, %1, %0) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
    ^2(%arg3_1 : i32, %arg4_1 : i32, %arg5_1 : i32):
      %5 = "arith.muli"(%arg3_1, %arg4_1) : (i32, i32) -> i32
      "linalg.yield"(%5) : (i32) -> ()
    }) : (memref<64xi32, "L1">, memref<64xi32, "L1">, memref<64xi32, "L1">) -> ()
    "memref.copy"(%2, %arg2) : (memref<64xi32, "L1">, memref<64xi32, "L3">) -> ()
    "memref.copy"(%1, %arg1) : (memref<64xi32, "L1">, memref<64xi32, "L3">) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   "func.func"() <{sym_name = "simple_mult", function_type = (memref<64xi32, "L3">, memref<64xi32, "L3">, memref<64xi32, "L3">) -> (), sym_visibility = "public"}> ({
//CHECK-NEXT:   ^0(%arg0 : memref<64xi32, "L3">, %arg1 : memref<64xi32, "L3">, %arg2 : memref<64xi32, "L3">):
//CHECK-NEXT:     %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>, alignment = 64 : i64}> : () -> memref<64xi32, "L1">
//CHECK-NEXT:     %1 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>, alignment = 64 : i64}> : () -> memref<64xi32, "L1">
//CHECK-NEXT:     %2 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>, alignment = 64 : i64}> : () -> memref<64xi32, "L1">
//CHECK-NEXT:     %3 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>, alignment = 64 : i64}> : () -> memref<64xi32, "L1">
//CHECK-NEXT:     "memref.copy"(%arg0, %0) : (memref<64xi32, "L3">, memref<64xi32, "L1">) -> ()
//CHECK-NEXT:     "memref.copy"(%arg1, %1) : (memref<64xi32, "L3">, memref<64xi32, "L1">) -> ()
//CHECK-NEXT:     "memref.copy"(%arg1, %3) : (memref<64xi32, "L3">, memref<64xi32, "L1">) -> ()
//CHECK-NEXT:     "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:     "linalg.generic"(%0, %1, %2) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
//CHECK-NEXT:     ^1(%arg3 : i32, %arg4 : i32, %arg5 : i32):
//CHECK-NEXT:       %4 = "arith.muli"(%arg3, %arg4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
//CHECK-NEXT:       "linalg.yield"(%4) : (i32) -> ()
//CHECK-NEXT:     }) : (memref<64xi32, "L1">, memref<64xi32, "L1">, memref<64xi32, "L1">) -> ()
//CHECK-NEXT:     "linalg.generic"(%3, %1, %0) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
//CHECK-NEXT:     ^2(%arg3_1 : i32, %arg4_1 : i32, %arg5_1 : i32):
//CHECK-NEXT:       %5 = "arith.muli"(%arg3_1, %arg4_1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
//CHECK-NEXT:       "linalg.yield"(%5) : (i32) -> ()
//CHECK-NEXT:     }) : (memref<64xi32, "L1">, memref<64xi32, "L1">, memref<64xi32, "L1">) -> ()
//CHECK-NEXT:     "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:     "memref.copy"(%2, %arg2) : (memref<64xi32, "L1">, memref<64xi32, "L3">) -> ()
//CHECK-NEXT:     "memref.copy"(%1, %arg1) : (memref<64xi32, "L1">, memref<64xi32, "L3">) -> ()
//CHECK-NEXT:     "func.return"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()
