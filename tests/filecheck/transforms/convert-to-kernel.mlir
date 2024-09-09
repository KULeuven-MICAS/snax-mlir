// RUN: ./compiler/snax-opt --split-input-file -p convert-to-kernel %s | filecheck %s

%0, %1, %2 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>)
linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
^0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
  %3 = arith.muli %arg0, %arg1 : i32
  linalg.yield %3 : i32
}

// CHECK: builtin.module {
// CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>)
// CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
// CHECK-NEXT:   ^0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
// CHECK-NEXT:     %3 = kernel.mul %arg0, %arg1 : i32, i32 -> i32
// CHECK-NEXT:     linalg.yield %3 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

%0, %1, %2 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>)
linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
^0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
  %3 = arith.addi %arg0, %arg1 : i32
  linalg.yield %3 : i32
}

// CHECK: builtin.module {
// CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>)
// CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
// CHECK-NEXT:   ^0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
// CHECK-NEXT:     %3 = kernel.add %arg0, %arg1 : i32, i32 -> i32
// CHECK-NEXT:     linalg.yield %3 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

%0, %1, %2 = "test.op"() : () -> (memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32>)
linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : memref<64x64xi32>, memref<64x64xi32>) outs(%2 : memref<64x64xi32>) {
^bb0(%in: i32, %in_0: i32, %out: i32):
  %3 = arith.muli %in, %in_0 : i32
  %4 = arith.addi %out, %3 : i32
  linalg.yield %4 : i32
}

// CHECK: builtin.module {
// CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32>)
// CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : memref<64x64xi32>, memref<64x64xi32>) outs(%2 : memref<64x64xi32>) {
// CHECK-NEXT:   ^0(%in : i32, %in_1 : i32, %out : i32):
// CHECK-NEXT:     %3 = kernel.mac %in, %in_1 : i32, i32 -> i32
// CHECK-NEXT:     linalg.yield %3 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

%0, %1, %2, %3 = "test.op"() : () -> (memref<?x?xi8>, memref<?x?xi8>, memref<?x?xi32>, i32)
linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1, %3, %3 : memref<?x?xi8>, memref<?x?xi8>, i32, i32) outs(%2 : memref<?x?xi32>) {
^bb0(%in: i8, %in_0: i8, %in_1: i32, %in_2: i32, %out: i32):
  %4 = arith.extsi %in : i8 to i32
  %5 = arith.subi %4, %in_1 : i32
  %6 = arith.extsi %in_0 : i8 to i32
  %7 = arith.subi %6, %in_2 : i32
  %8 = arith.muli %5, %7 : i32
  %9 = arith.addi %out, %8 : i32
  linalg.yield %9 : i32
}

// CHECK: builtin.module {
// CHECK-NEXT:   %0, %1, %2, %3 = "test.op"() : () -> (memref<?x?xi8>, memref<?x?xi8>, memref<?x?xi32>, i32)
// CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1, %3, %3 : memref<?x?xi8>, memref<?x?xi8>, i32, i32) outs(%2 : memref<?x?xi32>) {
// CHECK-NEXT:   ^0(%in : i8, %in_1 : i8, %in_2 : i32, %in_3 : i32, %out : i32):
// CHECK-NEXT:     %4 = kernel.qmac %in_2, %in_3 zp_lhs : %in zp_rhs : %in_1 : i32, i32, i8, i8 -> i32
// CHECK-NEXT:     linalg.yield %4 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }
