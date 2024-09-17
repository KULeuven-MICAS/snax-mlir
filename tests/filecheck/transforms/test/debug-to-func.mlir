// RUN: ./compiler/snax-opt %s -p test-debug-to-func | filecheck %s

builtin.module {
  func.func public @streamer_add(%arg0 : memref<?xi64>, %arg1 : memref<?xi64>, %arg2 : memref<?xi64>) {
    "debug.linalg"(%arg0, %arg1, %arg2) <{"debug_type" = "kernel_mac", "when" = "before", "level" = "none"}> : (memref<?xi64>, memref<?xi64>, memref<?xi64>) -> ()
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xi64>, memref<?xi64>) outs(%arg2 : memref<?xi64>) {
    ^0(%arg3 : i64, %arg4 : i64, %arg5 : i64):
      %0 = kernel.mac %arg3, %arg4 : i64, i64 -> i64
      linalg.yield %0 : i64
    }
    "debug.linalg"(%arg0, %arg1, %arg2) <{"debug_type" = "kernel_mac", "when" = "after", "level" = "none"}> : (memref<?xi64>, memref<?xi64>, memref<?xi64>) -> ()
    func.return
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:   func.func public @streamer_add(%arg0 : memref<?xi64>, %arg1 : memref<?xi64>, %arg2 : memref<?xi64>) {
// CHECK-NEXT:     %0 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi64>) -> index
// CHECK-NEXT:     %1 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi64>) -> index
// CHECK-NEXT:     %2 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi64>) -> index
// CHECK-NEXT:     %3 = arith.index_cast %0 : index to i32
// CHECK-NEXT:     %4 = arith.index_cast %1 : index to i32
// CHECK-NEXT:     %5 = arith.index_cast %2 : index to i32
// CHECK-NEXT:     %6 = arith.constant 5 : i32
// CHECK-NEXT:     func.call @debug_kernel_mac(%3, %4, %5, %6) : (i32, i32, i32, i32) -> ()
// CHECK-NEXT:     linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xi64>, memref<?xi64>) outs(%arg2 : memref<?xi64>) {
// CHECK-NEXT:     ^0(%arg3 : i64, %arg4 : i64, %arg5 : i64):
// CHECK-NEXT:       %7 = kernel.mac %arg3, %arg4 : i64, i64 -> i64
// CHECK-NEXT:       linalg.yield %7 : i64
// CHECK-NEXT:     }
// CHECK-NEXT:     %8 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi64>) -> index
// CHECK-NEXT:     %9 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi64>) -> index
// CHECK-NEXT:     %10 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi64>) -> index
// CHECK-NEXT:     %11 = arith.index_cast %8 : index to i32
// CHECK-NEXT:     %12 = arith.index_cast %9 : index to i32
// CHECK-NEXT:     %13 = arith.index_cast %10 : index to i32
// CHECK-NEXT:     %14 = arith.constant 5 : i32
// CHECK-NEXT:     func.call @debug_kernel_mac(%11, %12, %13, %14) : (i32, i32, i32, i32) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @debug_kernel_mac(i32, i32, i32, i32) -> ()
// CHECK-NEXT:  }
