// RUN: ./compiler/snax-opt %s -p dispatch-elementwise-mult --allow-unregistered-dialect --print-op-generic | filecheck %s

"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%A : memref<64xi32>, %B : memref<64xi32>, %D : memref<64xi32>):
    "linalg.mul"(%A, %B, %D) <{"operandSegmentSizes" = array<i32: 2, 1>}> ({
    ^1(%arg2 : i32, %arg3 : i32, %arg4 : i32):
      %0 = "arith.muli"(%arg2, %arg3) : (i32, i32) -> i32
      "linalg.yield"(%0) : (i32) -> ()
    }) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "public"}> ({
//CHECK-NEXT:   ^0(%A : memref<64xi32>, %B : memref<64xi32>, %D : memref<64xi32>):
//CHECK-NEXT:     "func.call"(%A, %B, %D) <{"callee" = @snax_hwpe_mult}> : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
//CHECK-NEXT:     "func.return"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snax_hwpe_mult", "function_type" = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), "sym_visibility" = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()