// RUN: ./compiler/snax-opt --split-input-file %s -p set-memory-layout --print-op-generic | filecheck %s

"builtin.module"() ({
  "func.func"() <{"sym_name" = "mnist", "function_type" = (memref<?x128xi8, 1 : i32>, memref<128x128xi8, 1 : i32>, memref<?x128xi32, 1 : i32>) -> memref<?x128xi32, 1 : i32>}> ({
  ^0(%arg0 : memref<?x128xi8, 1 : i32>, %arg1 : memref<128x128xi8, 1 : i32>, %arg2 : memref<?x128xi32, 1 : i32>):
    %0 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
    %1 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
    "linalg.generic"(%arg0, %arg1, %0, %1, %arg2) <{"indexing_maps" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], "iterator_types" = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], "operandSegmentSizes" = array<i32: 4, 1>}> ({
    ^1(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
      %2 = "arith.extsi"(%arg3) : (i8) -> i32
      %3 = "arith.subi"(%2, %arg5) : (i32, i32) -> i32
      %4 = "arith.extsi"(%arg4) : (i8) -> i32
      %5 = "arith.subi"(%4, %arg6) : (i32, i32) -> i32
      %6 = "arith.muli"(%3, %5) : (i32, i32) -> i32
      %7 = "arith.addi"(%arg7, %6) : (i32, i32) -> i32
      "linalg.yield"(%7) : (i32) -> ()
    }) {"library_call" = "snax_qgemm"} : (memref<?x128xi8, 1 : i32>, memref<128x128xi8, 1 : i32>, i32, i32, memref<?x128xi32, 1 : i32>) -> ()
    "func.return"(%arg2) : (memref<?x128xi32, 1 : i32>) -> ()
  }) : () -> ()
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   "func.func"() <{"sym_name" = "mnist", "function_type" = (memref<?x128xi8, 1 : i32>, memref<128x128xi8, 1 : i32>, memref<?x128xi32, 1 : i32>) -> memref<?x128xi32, 1 : i32>}> ({
// CHECK-NEXT:   ^0(%arg0 : memref<?x128xi8, 1 : i32>, %arg1 : memref<128x128xi8, 1 : i32>, %arg2 : memref<?x128xi32, 1 : i32>):
// CHECK-NEXT:     %0 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
// CHECK-NEXT:     %1 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
// CHECK-NEXT:     %2 = "snax.layout_cast"(%arg0) : (memref<?x128xi8, 1 : i32>) -> memref<?x128xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, 1 : i32>
// CHECK-NEXT:     %3 = "snax.layout_cast"(%arg1) : (memref<128x128xi8, 1 : i32>) -> memref<128x128xi8, #tsl.tsl<[?, 8] -> (256, 1), [?, 8] -> (?, 8), offset: 64>, 1 : i32>
// CHECK-NEXT:     %4 = "snax.layout_cast"(%arg2) : (memref<?x128xi32, 1 : i32>) -> memref<?x128xi32, #tsl.tsl<[?, 8] -> (256, 4), [?, 8] -> (?, 32)>, 1 : i32>
// CHECK-NEXT:     "linalg.generic"(%2, %3, %0, %1, %4) <{"indexing_maps" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], "iterator_types" = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], "operandSegmentSizes" = array<i32: 4, 1>}> ({
// CHECK-NEXT:     ^1(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
// CHECK-NEXT:       %5 = "arith.extsi"(%arg3) : (i8) -> i32
// CHECK-NEXT:       %6 = "arith.subi"(%5, %arg5) : (i32, i32) -> i32
// CHECK-NEXT:       %7 = "arith.extsi"(%arg4) : (i8) -> i32
// CHECK-NEXT:       %8 = "arith.subi"(%7, %arg6) : (i32, i32) -> i32
// CHECK-NEXT:       %9 = "arith.muli"(%6, %8) : (i32, i32) -> i32
// CHECK-NEXT:       %10 = "arith.addi"(%arg7, %9) : (i32, i32) -> i32
// CHECK-NEXT:       "linalg.yield"(%10) : (i32) -> ()
// CHECK-NEXT:     }) {"library_call" = "snax_qgemm"} : (memref<?x128xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, 1 : i32>, memref<128x128xi8, #tsl.tsl<[?, 8] -> (256, 1), [?, 8] -> (?, 8), offset: 64>, 1 : i32>, i32, i32, memref<?x128xi32, #tsl.tsl<[?, 8] -> (256, 4), [?, 8] -> (?, 32)>, 1 : i32>) -> ()
// CHECK-NEXT:     "func.return"(%arg2) : (memref<?x128xi32, 1 : i32>) -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()

