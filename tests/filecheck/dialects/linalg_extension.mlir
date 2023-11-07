// RUN: XDSL_ROUNDTRIP

%0, %1, %2 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>)

"linalg.mul"(%0, %1, %2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):
        %s1 = "arith.muli"(%arg2, %arg3) : (i32, i32) -> i32
        "linalg.yield"(%s1) : (i32) -> ()
    }) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()

//CHECK: builtin.module {
//CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>)
//CHECK-NEXT:   "linalg.mul"(%0, %1, %2) <{"operandSegmentSizes" = array<i32: 2, 1>}> ({
//CHECK-NEXT:   ^0(%arg2 : i32, %arg3 : i32, %arg4 : i32):
//CHECK-NEXT:     %s1 = arith.muli %arg2, %arg3 : i32
//CHECK-NEXT:     linalg.yield %s1 : i32
//CHECK-NEXT:   }) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
//CHECK-NEXT: }
