// RUN: ./compiler/snax-opt --split-input-file %s -p dispatch-regions --print-op-generic | filecheck %s

// test function without dispatchable ops
"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<64xi32>, %arg1 : memref<64xi32>, %arg2 : memref<64xi32>):
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "public"}> ({
//CHECK-NEXT:   ^0(%arg0 : memref<64xi32>, %arg1 : memref<64xi32>, %arg2 : memref<64xi32>):
//CHECK-NEXT:     "func.return"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()
// -----
// test function with dispatchable op to compute core
"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%0 : memref<64xi32>, %1 : memref<64xi32>, %2 : memref<64xi32>):
    "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
    ^1(%arg0 : i32, %arg1 : i32, %arg2 : i32):
      %3 = "arith.muli"(%arg0, %arg1) : (i32, i32) -> i32
      "linalg.yield"(%3) : (i32) -> ()
    }) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "public"}> ({
//CHECK-NEXT:   ^0(%0 : memref<64xi32>, %1 : memref<64xi32>, %2 : memref<64xi32>):
//CHECK-NEXT:     %3 = "func.call"() <{"callee" = @snrt_is_compute_core}> : () -> i1
//CHECK-NEXT:     "scf.if"(%3) ({
//CHECK-NEXT:       "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
//CHECK-NEXT:       ^1(%arg0 : i32, %arg1 : i32, %arg2 : i32):
//CHECK-NEXT:         %4 = "arith.muli"(%arg0, %arg1) : (i32, i32) -> i32
//CHECK-NEXT:         "linalg.yield"(%4) : (i32) -> ()
//CHECK-NEXT:       }) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
//CHECK-NEXT:       "scf.yield"() : () -> ()
//CHECK-NEXT:     }, {
//CHECK-NEXT:     }) : (i1) -> ()
//CHECK-NEXT:     "func.return"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snrt_is_compute_core", "function_type" = () -> i1, "sym_visibility" = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()
// -----
// test function with dispatchable op to dm core
"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%0 : memref<64xi32>, %1 : memref<64xi32>):
    "memref.copy"(%0, %1) : (memref<64xi32>, memref<64xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
//CHECK: "builtin.module"() ({
//CHECK-NEXT:   "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "public"}> ({
//CHECK-NEXT:   ^0(%0 : memref<64xi32>, %1 : memref<64xi32>):
//CHECK-NEXT:     %2 = "func.call"() <{"callee" = @snrt_is_dm_core}> : () -> i1
//CHECK-NEXT:     "scf.if"(%2) ({
//CHECK-NEXT:       "memref.copy"(%0, %1) : (memref<64xi32>, memref<64xi32>) -> ()
//CHECK-NEXT:       "scf.yield"() : () -> ()
//CHECK-NEXT:     }, {
//CHECK-NEXT:     }) : (i1) -> ()
//CHECK-NEXT:     "func.return"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snrt_is_dm_core", "function_type" = () -> i1, "sym_visibility" = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()
// -----
// test function with dispatchable ops to both cores
"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%0 : memref<64xi32>, %1 : memref<64xi32>, %2 : memref<64xi32>):
    "memref.copy"(%0, %1) : (memref<64xi32>, memref<64xi32>) -> ()
    "snax.cluster_sync_op"() : () -> ()
    "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
    ^1(%arg0 : i32, %arg1 : i32, %arg2 : i32):
      %3 = "arith.muli"(%arg0, %arg1) : (i32, i32) -> i32
      "linalg.yield"(%3) : (i32) -> ()
    }) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
//CHECK: "builtin.module"() ({
//CHECK-NEXT:   "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "public"}> ({
//CHECK-NEXT:   ^0(%0 : memref<64xi32>, %1 : memref<64xi32>, %2 : memref<64xi32>):
//CHECK-NEXT:     %3 = "func.call"() <{"callee" = @snrt_is_compute_core}> : () -> i1
//CHECK-NEXT:     %4 = "func.call"() <{"callee" = @snrt_is_dm_core}> : () -> i1
//CHECK-NEXT:     "scf.if"(%4) ({
//CHECK-NEXT:       "memref.copy"(%0, %1) : (memref<64xi32>, memref<64xi32>) -> ()
//CHECK-NEXT:       "scf.yield"() : () -> ()
//CHECK-NEXT:     }, {
//CHECK-NEXT:     }) : (i1) -> ()
//CHECK-NEXT:     "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:     "scf.if"(%3) ({
//CHECK-NEXT:       "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
//CHECK-NEXT:       ^1(%arg0 : i32, %arg1 : i32, %arg2 : i32):
//CHECK-NEXT:         %5 = "arith.muli"(%arg0, %arg1) : (i32, i32) -> i32
//CHECK-NEXT:         "linalg.yield"(%5) : (i32) -> ()
//CHECK-NEXT:       }) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
//CHECK-NEXT:       "scf.yield"() : () -> ()
//CHECK-NEXT:     }, {
//CHECK-NEXT:     }) : (i1) -> ()
//CHECK-NEXT:     "func.return"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snrt_is_dm_core", "function_type" = () -> i1, "sym_visibility" = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snrt_is_compute_core", "function_type" = () -> i1, "sym_visibility" = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()

