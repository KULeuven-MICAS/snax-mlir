// RUN: ./compiler/snax-opt --split-input-file %s -p dispatch-kernels --allow-unregistered-dialect --print-op-generic | filecheck %s

"builtin.module"() ({
  %0, %1, %2 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>)
  "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
  ^0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
    %3 = "arith.muli"(%arg0, %arg1) : (i32, i32) -> i32
    "linalg.yield"(%3) : (i32) -> ()
  }) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>)
//CHECK-NEXT:   "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
//CHECK-NEXT:   ^0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
//CHECK-NEXT:     %3 = "arith.muli"(%arg0, %arg1) : (i32, i32) -> i32
//CHECK-NEXT:     "linalg.yield"(%3) : (i32) -> ()
//CHECK-NEXT:   }) {"library_call" = "snax_hwpe_mult"} : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
//CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  "func.func"() <{function_type = (memref<?x128xi8>, memref<128x128xi8>, memref<?x128xi32>) -> memref<?x128xi32>, sym_name = "mnist"}> ({
  ^bb0(%arg0: memref<?x128xi8>, %arg1: memref<128x128xi8>, %arg2: memref<?x128xi32>):
    %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "linalg.generic"(%arg0, %arg1, %0, %1, %arg2) <{"indexing_maps" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], "iterator_types" = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 4, 1>}> ({
    ^bb0(%arg3: i8, %arg4: i8, %arg5: i32, %arg6: i32, %arg7: i32):
      %3 = "arith.extsi"(%arg3) : (i8) -> i32
      %4 = "arith.subi"(%3, %arg5) : (i32, i32) -> i32
      %5 = "arith.extsi"(%arg4) : (i8) -> i32
      %6 = "arith.subi"(%5, %arg6) : (i32, i32) -> i32
      %7 = "arith.muli"(%4, %6) : (i32, i32) -> i32
      %8 = "arith.addi"(%arg7, %7) : (i32, i32) -> i32
      "linalg.yield"(%8) : (i32) -> ()
    }) : (memref<?x128xi8>, memref<128x128xi8>, i32, i32, memref<?x128xi32>) -> ()
    "func.return"(%arg2) : (memref<?x128xi32>) -> ()
  }) : () -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   "func.func"() <{"function_type" = (memref<?x128xi8>, memref<128x128xi8>, memref<?x128xi32>) -> memref<?x128xi32>, "sym_name" = "mnist"}> ({
//CHECK-NEXT:   ^0(%arg0 : memref<?x128xi8>, %arg1 : memref<128x128xi8>, %arg2 : memref<?x128xi32>):
//CHECK-NEXT:     %0 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
//CHECK-NEXT:     %1 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
//CHECK-NEXT:     "linalg.generic"(%arg0, %arg1, %0, %1, %arg2) <{"indexing_maps" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], "iterator_types" = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], "operandSegmentSizes" = array<i32: 4, 1>}> ({
//CHECK-NEXT:     ^1(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
//CHECK-NEXT:       %2 = "arith.extsi"(%arg3) : (i8) -> i32
//CHECK-NEXT:       %3 = "arith.subi"(%2, %arg5) : (i32, i32) -> i32
//CHECK-NEXT:       %4 = "arith.extsi"(%arg4) : (i8) -> i32
//CHECK-NEXT:       %5 = "arith.subi"(%4, %arg6) : (i32, i32) -> i32
//CHECK-NEXT:       %6 = "arith.muli"(%3, %5) : (i32, i32) -> i32
//CHECK-NEXT:       %7 = "arith.addi"(%arg7, %6) : (i32, i32) -> i32
//CHECK-NEXT:       "linalg.yield"(%7) : (i32) -> ()
//CHECK-NEXT:     }) {"library_call" = "snax_qgemm"} : (memref<?x128xi8>, memref<128x128xi8>, i32, i32, memref<?x128xi32>) -> ()
//CHECK-NEXT:     "func.return"(%arg2) : (memref<?x128xi32>) -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()
