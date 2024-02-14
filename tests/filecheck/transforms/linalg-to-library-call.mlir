// RUN: ./compiler/snax-opt %s -p linalg-to-library-call --allow-unregistered-dialect --print-op-generic | filecheck %s

"builtin.module"() ({
  %0, %1, %2, %3 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32>)
  "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
  ^0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
    %4 = "arith.muli"(%arg0, %arg1) : (i32, i32) -> i32
    "linalg.yield"(%4) : (i32) -> ()
  }) {"library_call" = "snax_hwpe_mult"} : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
  "linalg.generic"(%1, %2, %3) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
  ^0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
    %5 = "arith.muli"(%arg0, %arg1) : (i32, i32) -> i32
    "linalg.yield"(%5) : (i32) -> ()
  }) {"library_call" = "snax_hwpe_mult"} : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   %0, %1, %2, %3 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32>)
//CHECK-NEXT:   %4 = "memref.cast"(%0) : (memref<64xi32>) -> memref<?xi32>
//CHECK-NEXT:   %5 = "memref.cast"(%1) : (memref<64xi32>) -> memref<?xi32>
//CHECK-NEXT:   %6 = "memref.cast"(%2) : (memref<64xi32>) -> memref<?xi32>
//CHECK-NEXT:   "func.call"(%4, %5, %6) <{"callee" = @snax_hwpe_mult}> : (memref<?xi32>, memref<?xi32>, memref<?xi32>) -> ()
//CHECK-NEXT:   %7 = "memref.cast"(%1) : (memref<64xi32>) -> memref<?xi32>
//CHECK-NEXT:   %8 = "memref.cast"(%2) : (memref<64xi32>) -> memref<?xi32>
//CHECK-NEXT:   %9 = "memref.cast"(%3) : (memref<64xi32>) -> memref<?xi32>
//CHECK-NEXT:   "func.call"(%7, %8, %9) <{"callee" = @snax_hwpe_mult}> : (memref<?xi32>, memref<?xi32>, memref<?xi32>) -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snax_hwpe_mult", "function_type" = (memref<?xi32>, memref<?xi32>, memref<?xi32>) -> (), "sym_visibility" = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()
