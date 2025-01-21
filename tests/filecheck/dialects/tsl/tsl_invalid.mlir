// RUN: XDSL_PARSING_DIAG


builtin.module {
  %0 = "test.op"() : () -> memref<64x64xindex, #tsl.tsl<[a, b] -> (8, 1), [16, 4] -> (256, 64), offset: 5>, 2 : i32>
}

// Expected an integer literal or ?
// CHECK: '>' expected


// -----

builtin.module {
  %0 = "test.op"() : () -> memref<64x64xindex, #tsl.tsl<8, 8] -> (8, 1), [16, 4] -> (256, 64), offset: 5>, 2 : i32>
}

//Expected opening bracket
// CHECK: '>' expected

// -----

builtin.module {
  %0 = "test.op"() : () -> memref<64x64xindex, #tsl.tsl<[8, 8] -> 8, 1), [16, 4] -> (256, 64), offset: 5>, 2 : i32>
}

//Expected opening bracket
// CHECK: '>' expected

// -----

builtin.module {
  %0 = "test.op"() : () -> memref<64x64xindex, #tsl.tsl<[8, 8] :) (8, 1), [16, 4] -> (256, 64), offset: 5>, 2 : i32>
}

//Expected arrow
// CHECK: '>' expected

// -----

builtin.module {
  %0 = "test.op"() : () -> memref<64x64xindex, #tsl.tsl<[8, 8] -> (64, 8, 1), [16, 4] -> (256, 64), offset: 5>, 2 : i32>
}

//Expected same number of strides and bounds
// CHECK: '>' expected

// -----

builtin.module {
  %0 = "test.op"() : () -> memref<64x64xindex, #tsl.tsl<[8, 8] -> (8, 1), [16, 4] -> (256, 64), offset] 5>, 2 : i32>
}

//Expected colon
// CHECK: '>' expected
