// RUN: ./compiler/snax-opt %s -p dispatch-regions{nb_cores=3\ pin_to_constants=true},function-constant-pinning,snax-to-func | mlir-opt-17 --canonicalize --inline| filecheck %s 

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

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func public @simple_mult(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = call @snax_cluster_core_idx() : () -> i32
// CHECK-NEXT:     %1 = arith.cmpi eq, %0, %c0_i32 : i32
// CHECK-NEXT:     scf.if %1 {
// CHECK-NEXT:       func.call @snax_cluster_hw_barrier() : () -> ()
// CHECK-NEXT:       linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xi32>, memref<64xi32>) outs(%arg2 : memref<64xi32>) {
// CHECK-NEXT:       ^bb0(%in: i32, %in_0: i32, %out: i32):
// CHECK-NEXT:         %2 = arith.muli %in, %in_0 : i32
// CHECK-NEXT:         linalg.yield %2 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %2 = arith.cmpi eq, %0, %c1_i32 : i32
// CHECK-NEXT:       scf.if %2 {
// CHECK-NEXT:         func.call @snax_cluster_hw_barrier() : () -> ()
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %3 = arith.cmpi eq, %0, %c2_i32 : i32
// CHECK-NEXT:         scf.if %3 {
// CHECK-NEXT:           memref.copy %arg0, %arg1 : memref<64xi32> to memref<64xi32>
// CHECK-NEXT:           func.call @snax_cluster_hw_barrier() : () -> ()
// CHECK-NEXT:         } else {
// CHECK-NEXT:           %4 = arith.cmpi eq, %0, %c0_i32 : i32
// CHECK-NEXT:           %5 = arith.cmpi eq, %0, %c2_i32 : i32
// CHECK-NEXT:           scf.if %5 {
// CHECK-NEXT:             memref.copy %arg0, %arg1 : memref<64xi32> to memref<64xi32>
// CHECK-NEXT:           }
// CHECK-NEXT:           func.call @snax_cluster_hw_barrier() : () -> ()
// CHECK-NEXT:           scf.if %4 {
// CHECK-NEXT:             linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xi32>, memref<64xi32>) outs(%arg2 : memref<64xi32>) {
// CHECK-NEXT:             ^bb0(%in: i32, %in_0: i32, %out: i32):
// CHECK-NEXT:               %6 = arith.muli %in, %in_0 : i32
// CHECK-NEXT:               linalg.yield %6 : i32
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func public @simple_mult_pinned_2(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
// CHECK-NEXT:     call @snax_cluster_hw_barrier() : () -> ()
// CHECK-NEXT:     linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xi32>, memref<64xi32>) outs(%arg2 : memref<64xi32>) {
// CHECK-NEXT:     ^bb0(%in: i32, %in_0: i32, %out: i32):
// CHECK-NEXT:       %0 = arith.muli %in, %in_0 : i32
// CHECK-NEXT:       linalg.yield %0 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func public @simple_mult_pinned_1(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
// CHECK-NEXT:     call @snax_cluster_hw_barrier() : () -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func public @simple_mult_pinned(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
// CHECK-NEXT:     memref.copy %arg0, %arg1 : memref<64xi32> to memref<64xi32>
// CHECK-NEXT:     call @snax_cluster_hw_barrier() : () -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @snax_cluster_core_idx() -> i32
// CHECK-NEXT:   func.func private @snax_cluster_hw_barrier()
// CHECK-NEXT: }


