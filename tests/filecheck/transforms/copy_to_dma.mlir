// RUN: ./compiler/snax-opt %s -p snax-copy-to-dma --print-op-generic | filecheck %s

"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<?xi32>, memref<?xi32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>):
    "memref.copy"(%arg0, %arg1) : (memref<?xi32>, memref<?xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<?xi32>, memref<?xi32>) -> (), "sym_visibility" = "public"}> ({
//CHECK-NEXT:   ^0(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>):
//CHECK-NEXT:     %0 = "arith.constant"() <{"value" = 0 : index}> : () -> index
//CHECK-NEXT:     %1 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index
//CHECK-NEXT:     %2 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
//CHECK-NEXT:     %3 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
//CHECK-NEXT:     "func.call"(%2, %3, %1) <{"callee" = @snax_dma_1d_transfer}> : (index, index, index) -> ()
//CHECK-NEXT:     "func.return"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snax_dma_1d_transfer", "function_type" = (index, index, index) -> (), "sym_visibility" = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()
