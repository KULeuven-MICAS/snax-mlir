// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_SINGLETRIP

//CHECK: module
"builtin.module"() ({
  //CHECK: #tsl.tsl<[8, 8] -> (8, 1), [16, 4] -> (256, 64), offset: 5>
  %0 = "test.op"() : () -> memref<64x64xindex, #tsl.tsl<[8, 8] -> (8, 1), [16, 4] -> (256, 64), offset: 5>, 2 : i32>
  //CHECK: #tsl.tsl<[?, 8] -> (32, 1), [?, 4] -> (?, 8), offset: 7>
  %1 = "test.op"() : () -> memref<?x?xindex, #tsl.tsl<[?, 8] -> (32, 1), [?, 4] -> (?, 8), offset: 7>, 2 : i32>
  //CHECK: #tsl.tsl<[8, 8] -> (8, 1), [16, 4] -> (256, 64), offset: 5>
  %2 = "test.op"() : () -> memref<64x64xindex, #tsl.tsl<[8, 8] -> (8, 1), [16, 4] -> (256, 64), offset: 5>, 2 : i32>
}) : () -> ()
