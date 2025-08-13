// RUN: XDSL_PARSING_DIAG


builtin.module {
  %0 = "test.op"() : () -> memref<64x64xindex, #tsl.tsl<[a, b] -> (8, 1), [16, 4] -> (256, 64), offset: 5>, 2 : i32>
}

// CHECK: Expected an integer literal or `?`


// -----

builtin.module {
  %0 = "test.op"() : () -> memref<64x64xindex, #tsl.tsl<8, 8] -> (8, 1), [16, 4] -> (256, 64), offset: 5>, 2 : i32>
}

// CHECK: Expected opening bracket

// -----

builtin.module {
  %0 = "test.op"() : () -> memref<64x64xindex, #tsl.tsl<[8, 8] -> 8, 1), [16, 4] -> (256, 64), offset: 5>, 2 : i32>
}

// CHECK: Expected opening bracket

// -----

builtin.module {
  %0 = "test.op"() : () -> memref<64x64xindex, #tsl.tsl<[8, 8] :) (8, 1), [16, 4] -> (256, 64), offset: 5>, 2 : i32>
}

// CHECK: Expected arrow

// -----

builtin.module {
  %0 = "test.op"() : () -> memref<64x64xindex, #tsl.tsl<[8, 8] -> (64, 8, 1), [16, 4] -> (256, 64), offset: 5>, 2 : i32>
}

// CHECK: Expected same number of steps and bounds

// -----

builtin.module {
  %0 = "test.op"() : () -> memref<64x64xindex, #tsl.tsl<[8, 8] -> (8, 1), [16, 4] -> (256, 64), offset] 5>, 2 : i32>
}

// CHECK: Expected colon
