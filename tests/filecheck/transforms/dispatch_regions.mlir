// RUN: ./compiler/snax-opt --split-input-file %s -p dispatch-regions --print-op-generic | filecheck %s --check-prefixes=CHECK,NB_TWO
// RUN: ./compiler/snax-opt --split-input-file %s -p dispatch-regions{nb_cores=3} --print-op-generic | filecheck %s --check-prefixes=CHECK,NB_THREE

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
//NB_TWO-NEXT:      %3 = "func.call"() <{"callee" = @snax_cluster_core_idx}> {"pin_to_constants" = [0 : i32, 1 : i32]} : () -> i32
//NB_THREE-NEXT:      %3 = "func.call"() <{"callee" = @snax_cluster_core_idx}> {"pin_to_constants" = [0 : i32, 1 : i32, 2 : i32]} : () -> i32
//CHECK-NEXT:     %4 = "arith.constant"() <{"value" = 0 : i32}> : () -> i32
//CHECK-NEXT:     %5 = "arith.cmpi"(%3, %4) <{"predicate" = 0 : i64}> : (i32, i32) -> i1
//CHECK-NEXT:     "scf.if"(%5) ({
//CHECK-NEXT:       "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
//CHECK-NEXT:       ^1(%arg0 : i32, %arg1 : i32, %arg2 : i32):
//CHECK-NEXT:         %6 = "arith.muli"(%arg0, %arg1) : (i32, i32) -> i32
//CHECK-NEXT:         "linalg.yield"(%6) : (i32) -> ()
//CHECK-NEXT:       }) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
//CHECK-NEXT:       "scf.yield"() : () -> ()
//CHECK-NEXT:     }, {
//CHECK-NEXT:     }) : (i1) -> ()
//CHECK-NEXT:     "func.return"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snax_cluster_core_idx", "function_type" = () -> i32, "sym_visibility" = "private"}> ({
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
//NB_TWO-NEXT:     %2 = "func.call"() <{"callee" = @snax_cluster_core_idx}> {"pin_to_constants" = [0 : i32, 1 : i32]} : () -> i32
//NB_TWO-NEXT:     %3 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
//NB_THREE-NEXT:     %2 = "func.call"() <{"callee" = @snax_cluster_core_idx}> {"pin_to_constants" = [0 : i32, 1 : i32, 2 : i32]} : () -> i32
//NB_THREE-NEXT:     %3 = "arith.constant"() <{"value" = 2 : i32}> : () -> i32
//CHECK-NEXT:     %4 = "arith.cmpi"(%2, %3) <{"predicate" = 0 : i64}> : (i32, i32) -> i1
//CHECK-NEXT:     "scf.if"(%4) ({
//CHECK-NEXT:       "memref.copy"(%0, %1) : (memref<64xi32>, memref<64xi32>) -> ()
//CHECK-NEXT:       "scf.yield"() : () -> ()
//CHECK-NEXT:     }, {
//CHECK-NEXT:     }) : (i1) -> ()
//CHECK-NEXT:     "func.return"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snax_cluster_core_idx", "function_type" = () -> i32, "sym_visibility" = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()
// -----
// test function with dispatchable ops to both cores
"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%0 : memref<64xi32>, %1 : memref<64xi32>, %2 : memref<64xi32>):
    %alloc = "memref.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0>}> {"alignment" = 64 : i64} : () -> memref<64xi32>
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
//NB_TWO-NEXT:     %3 = "func.call"() <{"callee" = @snax_cluster_core_idx}> {"pin_to_constants" = [0 : i32, 1 : i32]} : () -> i32
//NB_THREE-NEXT:     %3 = "func.call"() <{"callee" = @snax_cluster_core_idx}> {"pin_to_constants" = [0 : i32, 1 : i32, 2 : i32]} : () -> i32
//CHECK-NEXT:     %4 = "arith.constant"() <{"value" = 0 : i32}> : () -> i32
//CHECK-NEXT:     %5 = "arith.cmpi"(%3, %4) <{"predicate" = 0 : i64}> : (i32, i32) -> i1
//NB_TWO-NEXT:     %6 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
//NB_THREE-NEXT:     %6 = "arith.constant"() <{"value" = 2 : i32}> : () -> i32
//CHECK-NEXT:     %7 = "arith.cmpi"(%3, %6) <{"predicate" = 0 : i64}> : (i32, i32) -> i1
//CHECK-NEXT:     %alloc = "memref.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0>}> {"alignment" = 64 : i64} : () -> memref<64xi32>
//CHECK-NEXT:     "scf.if"(%7) ({
//CHECK-NEXT:       "memref.copy"(%0, %1) : (memref<64xi32>, memref<64xi32>) -> ()
//CHECK-NEXT:       "scf.yield"() : () -> ()
//CHECK-NEXT:     }, {
//CHECK-NEXT:     }) : (i1) -> ()
//CHECK-NEXT:     "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:     "scf.if"(%5) ({
//CHECK-NEXT:       "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
//CHECK-NEXT:       ^1(%arg0 : i32, %arg1 : i32, %arg2 : i32):
//CHECK-NEXT:         %8 = "arith.muli"(%arg0, %arg1) : (i32, i32) -> i32
//CHECK-NEXT:         "linalg.yield"(%8) : (i32) -> ()
//CHECK-NEXT:       }) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
//CHECK-NEXT:       "scf.yield"() : () -> ()
//CHECK-NEXT:     }, {
//CHECK-NEXT:     }) : (i1) -> ()
//CHECK-NEXT:     "func.return"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snax_cluster_core_idx", "function_type" = () -> i32, "sym_visibility" = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()
// -----
// test dispatchable region in nested if-statements
"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%0 : memref<64xi32>, %1 : memref<64xi32>, %2 : memref<64xi32>):
    %9 = arith.constant 1 : i1
    "scf.if"(%9) ({
      "scf.if"(%9) ({
        "scf.if"(%9) ({
          "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
            ^1(%arg0 : i32, %arg1 : i32, %arg2 : i32):
              %3 = "arith.muli"(%arg0, %arg1) : (i32, i32) -> i32
              "linalg.yield"(%3) : (i32) -> ()
          }) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
          "scf.yield"() : () -> ()
        }, {
        }) : (i1) -> ()
        "scf.yield"() : () -> ()
      }, {
      }) : (i1) -> ()
      "scf.yield"() : () -> ()
    }, {
    }) : (i1) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()


//CHECK: "builtin.module"() ({
//CHECK-NEXT:   "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "public"}> ({
//CHECK-NEXT:   ^0(%0 : memref<64xi32>, %1 : memref<64xi32>, %2 : memref<64xi32>):
//NB_TWO-NEXT:     %3 = "func.call"() <{"callee" = @snax_cluster_core_idx}> {"pin_to_constants" = [0 : i32, 1 : i32]} : () -> i32
//NB_THREE-NEXT:     %3 = "func.call"() <{"callee" = @snax_cluster_core_idx}> {"pin_to_constants" = [0 : i32, 1 : i32, 2 : i32]} : () -> i32
//CHECK-NEXT:     %4 = "arith.constant"() <{"value" = 0 : i32}> : () -> i32
//CHECK-NEXT:     %5 = "arith.cmpi"(%3, %4) <{"predicate" = 0 : i64}> : (i32, i32) -> i1
//CHECK-NEXT:     %6 = "arith.constant"() <{"value" = true}> : () -> i1
//CHECK-NEXT:     "scf.if"(%6) ({
//CHECK-NEXT:       "scf.if"(%6) ({
//CHECK-NEXT:         "scf.if"(%6) ({
//CHECK-NEXT:           "scf.if"(%5) ({
//CHECK-NEXT:             "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
//CHECK-NEXT:             ^1(%arg0 : i32, %arg1 : i32, %arg2 : i32):
//CHECK-NEXT:               %7 = "arith.muli"(%arg0, %arg1) : (i32, i32) -> i32
//CHECK-NEXT:               "linalg.yield"(%7) : (i32) -> ()
//CHECK-NEXT:             }) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
//CHECK-NEXT:             "scf.yield"() : () -> ()
//CHECK-NEXT:           }, {
//CHECK-NEXT:           }) : (i1) -> ()
//CHECK-NEXT:           "scf.yield"() : () -> ()
//CHECK-NEXT:         }, {
//CHECK-NEXT:         }) : (i1) -> ()
//CHECK-NEXT:         "scf.yield"() : () -> ()
//CHECK-NEXT:       }, {
//CHECK-NEXT:       }) : (i1) -> ()
//CHECK-NEXT:       "scf.yield"() : () -> ()
//CHECK-NEXT:     }, {
//CHECK-NEXT:     }) : (i1) -> ()
//CHECK-NEXT:     "func.return"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snax_cluster_core_idx", "function_type" = () -> i32, "sym_visibility" = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()


