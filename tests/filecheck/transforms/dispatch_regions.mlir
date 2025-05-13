// RUN: snax-opt --split-input-file %s -p dispatch-regions | filecheck %s --check-prefixes=CHECK,NB_TWO
// RUN: snax-opt --split-input-file %s -p dispatch-regions{nb_cores=3} | filecheck %s --check-prefixes=CHECK,NB_THREE

// test function without dispatchable ops
"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<64xi32>, %arg1 : memref<64xi32>, %arg2 : memref<64xi32>):
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%arg0 : memref<64xi32>, %arg1 : memref<64xi32>, %arg2 : memref<64xi32>) {
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

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

// CHECK: builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%0 : memref<64xi32>, %1 : memref<64xi32>, %2 : memref<64xi32>) {
// NB_TWO-NEXT:    %3 = func.call @snax_cluster_core_idx() {pin_to_constants = [0 : i32, 1 : i32]} : () -> i32
// NB_THREE-NEXT:  %3 = func.call @snax_cluster_core_idx() {pin_to_constants = [0 : i32, 1 : i32, 2 : i32]} : () -> i32
// CHECK-NEXT:     %4 = arith.constant 0 : i32
// CHECK-NEXT:     %5 = arith.cmpi eq, %3, %4 : i32
// CHECK-NEXT:     scf.if %5 {
// CHECK-NEXT:       linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
// CHECK-NEXT:       ^0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
// CHECK-NEXT:         %6 = arith.muli %arg0, %arg1 : i32
// CHECK-NEXT:         linalg.yield %6 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @snax_cluster_core_idx() -> i32
// CHECK-NEXT: }

// -----

// test function with dispatchable op to dm core
"builtin.module"() ({
  "func.func"() <{sym_name = "simple_mult", "function_type" = (memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%0 : memref<64xi32>, %1 : memref<64xi32>):
    "memref.copy"(%0, %1) : (memref<64xi32>, memref<64xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK: builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%0 : memref<64xi32>, %1 : memref<64xi32>) {
// NB_TWO-NEXT:    %2 = func.call @snax_cluster_core_idx() {pin_to_constants = [0 : i32, 1 : i32]} : () -> i32
// NB_TWO-NEXT:    %3 = arith.constant 1 : i32
// NB_THREE-NEXT:  %2 = func.call @snax_cluster_core_idx() {pin_to_constants = [0 : i32, 1 : i32, 2 : i32]} : () -> i32
// NB_THREE-NEXT:  %3 = arith.constant 2 : i32
// CHECK-NEXT:     %4 = arith.cmpi eq, %2, %3 : i32
// CHECK-NEXT:     scf.if %4 {
// CHECK-NEXT:       "memref.copy"(%0, %1) : (memref<64xi32>, memref<64xi32>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @snax_cluster_core_idx() -> i32
// CHECK-NEXT: }

// -----

// two copies after each other

func.func public @func() {
	%0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>
	"memref.copy"(%0, %0) : (memref<16x16xi8>, memref<16x16xi8>) -> ()
	"memref.copy"(%0, %0) : (memref<16x16xi8>, memref<16x16xi8>) -> ()
	"snax.cluster_sync_op"() : () -> ()
	func.return
}

// CHECK: 		builtin.module {
// CHECK-NEXT:   func.func public @func() {
// CHECK-NEXT:     %0 = func.call @snax_cluster_core_idx() {pin_to_constants =
// CHECK-NEXT:     %1 = arith.constant
// CHECK-NEXT:     %2 = arith.cmpi eq, %0, %1 : i32
// CHECK-NEXT:     %3 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>
// CHECK-NEXT:     scf.if %2 {
// CHECK-NEXT:       "memref.copy"(%3, %3) : (memref<16x16xi8>, memref<16x16xi8>) -> ()
// CHECK-NEXT:       "memref.copy"(%3, %3) : (memref<16x16xi8>, memref<16x16xi8>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     "snax.cluster_sync_op"() : () -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @snax_cluster_core_idx() -> i32
// CHECK-NEXT: }

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

// CHECK: builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%0 : memref<64xi32>, %1 : memref<64xi32>, %2 : memref<64xi32>) {
// CHECK-NEXT:     %3 = func.call @snax_cluster_core_idx() {pin_to_constants =
// CHECK-NEXT:     %4 = arith.constant 0 : i32
// CHECK-NEXT:     %5 = arith.cmpi eq, %3, %4 : i32
// CHECK-NEXT:     %6 = arith.constant
// CHECK-NEXT:     %7 = arith.cmpi eq, %3, %6 : i32
// CHECK-NEXT:     %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xi32>
// CHECK-NEXT:     scf.if %7 {
// CHECK-NEXT:       "memref.copy"(%0, %1) : (memref<64xi32>, memref<64xi32>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     "snax.cluster_sync_op"() : () -> ()
// CHECK-NEXT:     scf.if %5 {
// CHECK-NEXT:       linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
// CHECK-NEXT:       ^0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
// CHECK-NEXT:         %8 = arith.muli %arg0, %arg1 : i32
// CHECK-NEXT:         linalg.yield %8 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @snax_cluster_core_idx() -> i32
// CHECK-NEXT: }

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


// CHECK: builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%0 : memref<64xi32>, %1 : memref<64xi32>, %2 : memref<64xi32>) {
// CHECK-NEXT:     %3 = func.call @snax_cluster_core_idx() {pin_to_constants =
// CHECK-NEXT:     %4 = arith.constant 0 : i32
// CHECK-NEXT:     %5 = arith.cmpi eq, %3, %4 : i32
// CHECK-NEXT:     %6 = arith.constant true
// CHECK-NEXT:     scf.if %6 {
// CHECK-NEXT:       scf.if %6 {
// CHECK-NEXT:         scf.if %6 {
// CHECK-NEXT:           scf.if %5 {
// CHECK-NEXT:             linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
// CHECK-NEXT:             ^0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
// CHECK-NEXT:               %7 = arith.muli %arg0, %arg1 : i32
// CHECK-NEXT:               linalg.yield %7 : i32
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @snax_cluster_core_idx() -> i32
// CHECK-NEXT: }

// -----

func.func public @func() {
	%0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>
	%1 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>
	"memref.copy"(%0, %1) : (memref<16x16xi8>, memref<16x16xi8>) -> ()
	"test.op"() ({
		"test.op"() : () -> ()
		"test.termop"() : () -> ()
	}): () -> ()
	func.return
}

// CHECK: builtin.module {
// CHECK-NEXT:   func.func public @func() {
// CHECK-NEXT:     %0 = func.call @snax_cluster_core_idx() {pin_to_constants = [{{[^\]]*}}]} : () -> i32
// CHECK-NEXT:     %1 = arith.constant {{\d}} : i32
// CHECK-NEXT:     %2 = arith.cmpi eq, %0, %1 : i32
// CHECK-NEXT:     %3 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>
// CHECK-NEXT:     %4 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>
// CHECK-NEXT:     scf.if %2 {
// CHECK-NEXT:       "memref.copy"(%3, %4) : (memref<16x16xi8>, memref<16x16xi8>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     "test.op"() ({
// CHECK-NEXT:       "test.op"() : () -> ()
// CHECK-NEXT:       "test.termop"() : () -> ()
// CHECK-NEXT:     }) : () -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @snax_cluster_core_idx() -> i32
// CHECK-NEXT: }

// -----

func.func public @func() {
	%0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>
	%1 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>
	"memref.copy"(%0, %1) : (memref<16x16xi8>, memref<16x16xi8>) -> ()
	"test.op"() ({
		"memref.copy"(%0, %1) : (memref<16x16xi8>, memref<16x16xi8>) -> ()
		"test.op"() : () -> ()
		"test.termop"() : () -> ()
	}): () -> ()
	func.return
}

// CHECK: builtin.module {
// CHECK-NEXT:   func.func public @func() {
// CHECK-NEXT:     %0 = func.call @snax_cluster_core_idx() {pin_to_constants =
// CHECK-NEXT:     %1 = arith.constant
// CHECK-NEXT:     %2 = arith.cmpi eq, %0, %1 : i32
// CHECK-NEXT:     %3 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>
// CHECK-NEXT:     %4 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>
// CHECK-NEXT:     scf.if %2 {
// CHECK-NEXT:       "memref.copy"(%3, %4) : (memref<16x16xi8>, memref<16x16xi8>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     "test.op"() ({
// CHECK-NEXT:       scf.if %2 {
// CHECK-NEXT:         "memref.copy"(%3, %4) : (memref<16x16xi8>, memref<16x16xi8>) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:       "test.op"() : () -> ()
// CHECK-NEXT:       "test.termop"() : () -> ()
// CHECK-NEXT:     }) : () -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @snax_cluster_core_idx() -> i32
// CHECK-NEXT: }
