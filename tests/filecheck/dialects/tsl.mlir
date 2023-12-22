// RUN: XDSL_ROUNDTRIP

//CHECK: module
"builtin.module"() ({
  //CHECK: #tsl.tsl<[8, 8] -> (8, 1), [16, 4] -> (256, 64), offset: 5>
  %0 = "memref.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64x64xindex, #tsl.tsl<[8, 8] -> (8, 1), [16, 4] -> (256, 64), offset: 5>, 2 : i32>
  %1, %2 = "test.op"() : () -> (index, index)
  //CHECK: #tsl.tsl<[?, 8] -> (32, 1), [?, 4] -> (?, 8), offset: 7>
  %3 = "memref.alloc"(%1, %2) <{"operandSegmentSizes" = array<i32: 2, 0>}> : (index, index) -> memref<?x?xindex, #tsl.tsl<[?, 8] -> (32, 1), [?, 4] -> (?, 8), offset: 7>, 2 : i32>
}) : () -> ()
=======
  //CHECK: #tsl.tsl<([2, 4] * [4, 2], [16, 32] * [4, 2])>
  %0 = "memref.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64x64xindex, #tsl.tsl<([2, 4] * [4, 2], [16, 32] * [4, 2])>, 2 : i32>
}) : () -> ()
>>>>>>> ec209b9 (add simple filecheck)
