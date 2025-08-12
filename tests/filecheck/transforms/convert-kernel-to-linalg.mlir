// RUN: snax-opt --split-input-file -p convert-kernel-to-linalg %s | filecheck %s

%0, %1, %2 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>)
linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
^bb0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
  %3 = kernel.mul %arg0, %arg1 : i32, i32 -> i32
  linalg.yield %3 : i32
}

// CHECK: builtin.module {
// CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>)
// CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
// CHECK-NEXT:   ^bb0(%3 : i32, %4 : i32, %5 : i32):
// CHECK-NEXT:     %6 = arith.muli %3, %4 : i32
// CHECK-NEXT:     linalg.yield %6 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

%0, %1, %2 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>)
linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
^bb0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
  %3 = kernel.add %arg0, %arg1 : i32, i32 -> i32
  linalg.yield %3 : i32
}

// CHECK: builtin.module {
// CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>)
// CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
// CHECK-NEXT:   ^bb0(%3 : i32, %4 : i32, %5 : i32):
// CHECK-NEXT:     %6 = arith.addi %3, %4 : i32
// CHECK-NEXT:     linalg.yield %6 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

%0, %1, %2 = "test.op"() : () -> (memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32>)
linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : memref<64x64xi32>, memref<64x64xi32>) outs(%2 : memref<64x64xi32>) {
^bb0(%in : i32, %in_1 : i32, %out : i32):
  %3 = kernel.mac %in, %in_1 : i32, i32 -> i32
  linalg.yield %3 : i32
}

// CHECK: builtin.module {
// CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32>)
// CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : memref<64x64xi32>, memref<64x64xi32>) outs(%2 : memref<64x64xi32>) {
// CHECK-NEXT:   ^bb0(%3 : i32, %4 : i32, %5 : i32):
// CHECK-NEXT:     %6 = arith.muli %3, %4 : i32
// CHECK-NEXT:     %7 = arith.addi %5, %6 : i32
// CHECK-NEXT:     linalg.yield %7 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

%0, %1, %2, %3 = "test.op"() : () -> (memref<?x?xi8>, memref<?x?xi8>, memref<?x?xi32>, i32)
linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1, %3, %3 : memref<?x?xi8>, memref<?x?xi8>, i32, i32) outs(%2 : memref<?x?xi32>) {
^bb0(%in : i8, %in_1 : i8, %in_2 : i32, %in_3 : i32, %out : i32):
  %4 = kernel.qmac %in, %in_1 zp_lhs : %in_2 zp_rhs : %in_3 : i8, i8, i32, i32 -> i32
  linalg.yield %4 : i32
}

// CHECK: builtin.module {
// CHECK-NEXT:   %0, %1, %2, %3 = "test.op"() : () -> (memref<?x?xi8>, memref<?x?xi8>, memref<?x?xi32>, i32)
// CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1, %3, %3 : memref<?x?xi8>, memref<?x?xi8>, i32, i32) outs(%2 : memref<?x?xi32>) {
// CHECK-NEXT:   ^bb0(%4 : i8, %5 : i8, %6 : i32, %7 : i32, %8 : i32):
// CHECK-NEXT:     %9 = arith.extsi %4 : i8 to i32
// CHECK-NEXT:     %10 = arith.subi %9, %6 : i32
// CHECK-NEXT:     %11 = arith.extsi %5 : i8 to i32
// CHECK-NEXT:     %12 = arith.subi %11, %7 : i32
// CHECK-NEXT:     %13 = arith.muli %10, %12 : i32
// CHECK-NEXT:     %14 = arith.addi %8, %13 : i32
// CHECK-NEXT:     linalg.yield %14 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }
