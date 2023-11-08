// RUN: ./compiler/snax-opt %s -p linalg-to-library-call --allow-unregistered-dialect --print-op-generic | filecheck %s

"builtin.module"() ({
  %0, %1, %2 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>)
  "linalg.generic"(%0, %1, %2) <{"indexing_maps" = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
  ^0(%arg0 : i32, %arg1 : i32, %arg2 : i32):
    %3 = "arith.muli"(%arg0, %arg1) : (i32, i32) -> i32
    "linalg.yield"(%3) : (i32) -> ()
  }) {"library_call" = "snax_hwpe_mult"} : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
}) : () -> ()


//CHECK: "builtin.module"() ({
//CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>)
//CHECK-NEXT:   "func.call"(%0, %1, %2) <{"callee" = @snax_hwpe_mult}> : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snax_hwpe_mult", "function_type" = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()
