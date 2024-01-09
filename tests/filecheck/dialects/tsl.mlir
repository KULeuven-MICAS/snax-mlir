// RUN: XDSL_ROUNDTRIP

//CHECK: module
"builtin.module"() ({
  //CHECK: #tsl.tsl<([2, 4] * [4, 2], [16, 32] * [4, 2], offset: 5)>
  %0 = "memref.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64x64xindex, #tsl.tsl<([2, 4] * [4, 2], [16, 32] * [4, 2], offset: 5)>, 2 : i32>
  //CHECK: #tsl.tsl<([2, 4] * [4, ?], [16, ?] * [4, ?], offset: 7)>
  %1 = "memref.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64x64xindex, #tsl.tsl<([2, 4] * [4, ?], [16, ?] * [4, ?], offset: 7)>, 2 : i32>
}) : () -> ()
