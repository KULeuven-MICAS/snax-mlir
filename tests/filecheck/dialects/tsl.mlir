// RUN: XDSL_ROUNDTRIP

//CHECK: module
"builtin.module"() ({
  //CHECK: #tsl.tsl<([2, 4] * [4, 2], [16, 32] * [4, 2])>
  %0 = "memref.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64x64xindex, #tsl.tsl<([2, 4] * [4, 2], [16, 32] * [4, 2])>, 2 : i32>
}) : () -> ()