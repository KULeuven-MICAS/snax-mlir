// RUN: ./compiler/snax-opt %s -p test-insert-debugs | filecheck %s

func.func public @streamer_add(%arg0 : memref<?xi64>, %arg1 : memref<?xi64>, %arg2 : memref<?xi64>) {
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xi64>, memref<?xi64>) outs(%arg2 : memref<?xi64>) {
  ^0(%arg3 : i64, %arg4 : i64, %arg5 : i64):
    %0 = kernel.mac %arg3, %arg4 : i64, i64 -> i64
    linalg.yield %0 : i64
  }
  func.return
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func public @streamer_add(%arg0 : memref<?xi64>, %arg1 : memref<?xi64>, %arg2 : memref<?xi64>) {
// CHECK-NEXT:     "debug.debug"(%arg0, %arg1, %arg2) <{"debug_type" = "kernel_mac", "when" = "before", "level" = "none"}> : (memref<?xi64>, memref<?xi64>, memref<?xi64>) -> ()
// CHECK-NEXT:     linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xi64>, memref<?xi64>) outs(%arg2 : memref<?xi64>) {
// CHECK-NEXT:     ^0(%arg3 : i64, %arg4 : i64, %arg5 : i64):
// CHECK-NEXT:       %0 = kernel.mac %arg3, %arg4 : i64, i64 -> i64
// CHECK-NEXT:       linalg.yield %0 : i64
// CHECK-NEXT:     }
// CHECK-NEXT:     "debug.debug"(%arg0, %arg1, %arg2) <{"debug_type" = "kernel_mac", "when" = "after", "level" = "none"}> : (memref<?xi64>, memref<?xi64>, memref<?xi64>) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
