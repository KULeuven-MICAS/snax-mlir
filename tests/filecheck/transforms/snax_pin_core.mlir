// RUN: snax-opt %s -p function-constant-pinning | filecheck %s

builtin.module {
  func.func public @simple_mult(%0 : memref<64xi32>, %1 : memref<64xi32>, %2 : memref<64xi32>) {
    %3 = func.call @snax_cluster_core_idx() {"pin_to_constants" = [0 : i32, 1 : i32]} : () -> i32
    %4 = arith.constant 0 : i32
    %5 = arith.cmpi eq, %3, %4 : i32
    %6 = arith.constant 1 : i32
    %7 = arith.cmpi eq, %3, %6 : i32
    %alloc = memref.alloc() {"alignment" = 64 : i64} : memref<64xi32>
    "scf.if"(%7) ({
      "memref.copy"(%0, %1) : (memref<64xi32>, memref<64xi32>) -> ()
      scf.yield
    }, {
    }) : (i1) -> ()
    "snax.cluster_sync_op"() : () -> ()
    "scf.if"(%5) ({
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
      ^bb0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
        %8 = arith.muli %arg0, %arg1 : i32
        linalg.yield %8 : i32
      }
      scf.yield
    }, {
    }) : (i1) -> ()
    func.return
  }
  func.func private @snax_cluster_core_idx() -> i32
}

// CHECK: builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%0 : memref<64xi32>, %1 : memref<64xi32>, %2 : memref<64xi32>) {
// CHECK-NEXT:     %3 = func.call @snax_cluster_core_idx() : () -> i32
// CHECK-NEXT:     %4 = arith.constant 0 : i32
// CHECK-NEXT:     %5 = arith.cmpi eq, %3, %4 : i32
// CHECK-NEXT:     scf.if %5 {
// CHECK-NEXT:       func.call @simple_mult_pinned_1(%0, %1, %2) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %6 = arith.constant 1 : i32
// CHECK-NEXT:       %7 = arith.cmpi eq, %3, %6 : i32
// CHECK-NEXT:       scf.if %7 {
// CHECK-NEXT:         func.call @simple_mult_pinned(%0, %1, %2) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %8 = arith.constant 0 : i32
// CHECK-NEXT:         %9 = arith.cmpi eq, %3, %8 : i32
// CHECK-NEXT:         %10 = arith.constant 1 : i32
// CHECK-NEXT:         %11 = arith.cmpi eq, %3, %10 : i32
// CHECK-NEXT:         %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xi32>
// CHECK-NEXT:         scf.if %11 {
// CHECK-NEXT:           "memref.copy"(%0, %1) : (memref<64xi32>, memref<64xi32>) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:         "snax.cluster_sync_op"() : () -> ()
// CHECK-NEXT:         scf.if %9 {
// CHECK-NEXT:           linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
// CHECK-NEXT:           ^bb0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
// CHECK-NEXT:             %12 = arith.muli %arg0, %arg1 : i32
// CHECK-NEXT:             linalg.yield %12 : i32
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func public @simple_mult_pinned_1(%0 : memref<64xi32>, %1 : memref<64xi32>, %2 : memref<64xi32>) {
// CHECK-NEXT:     %3 = arith.constant 0 : i32
// CHECK-NEXT:     %4 = arith.constant 1 : i32
// CHECK-NEXT:     %5 = arith.cmpi eq, %3, %4 : i32
// CHECK-NEXT:     scf.if %5 {
// CHECK-NEXT:       func.call @simple_mult_pinned(%0, %1, %2) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %6 = arith.constant 0 : i32
// CHECK-NEXT:       %7 = arith.cmpi eq, %3, %6 : i32
// CHECK-NEXT:       %8 = arith.constant 1 : i32
// CHECK-NEXT:       %9 = arith.cmpi eq, %3, %8 : i32
// CHECK-NEXT:       %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xi32>
// CHECK-NEXT:       scf.if %9 {
// CHECK-NEXT:         "memref.copy"(%0, %1) : (memref<64xi32>, memref<64xi32>) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
// CHECK-NEXT:       scf.if %7 {
// CHECK-NEXT:         linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
// CHECK-NEXT:         ^bb0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
// CHECK-NEXT:           %10 = arith.muli %arg0, %arg1 : i32
// CHECK-NEXT:           linalg.yield %10 : i32
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func public @simple_mult_pinned(%0 : memref<64xi32>, %1 : memref<64xi32>, %2 : memref<64xi32>) {
// CHECK-NEXT:     %3 = arith.constant 1 : i32
// CHECK-NEXT:     %4 = arith.constant 0 : i32
// CHECK-NEXT:     %5 = arith.cmpi eq, %3, %4 : i32
// CHECK-NEXT:     %6 = arith.constant 1 : i32
// CHECK-NEXT:     %7 = arith.cmpi eq, %3, %6 : i32
// CHECK-NEXT:     %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xi32>
// CHECK-NEXT:     scf.if %7 {
// CHECK-NEXT:       "memref.copy"(%0, %1) : (memref<64xi32>, memref<64xi32>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     "snax.cluster_sync_op"() : () -> ()
// CHECK-NEXT:     scf.if %5 {
// CHECK-NEXT:       linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
// CHECK-NEXT:       ^bb0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
// CHECK-NEXT:         %8 = arith.muli %arg0, %arg1 : i32
// CHECK-NEXT:         linalg.yield %8 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @snax_cluster_core_idx() -> i32
// CHECK-NEXT: }
