builtin.module {
  %0, %1, %2, %3 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32>)
  memref_stream.streaming_region {bounds = [64], indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
  ^0(%4 : !stream.readable<i32>, %5 : !stream.readable<i32>, %6 : !stream.writable<i32>):
    memref_stream.generic {bounds = [#builtin.int<64>], indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : !stream.readable<i32>, !stream.readable<i32>) outs(%6 : !stream.writable<i32>) {
    ^1(%arg0 : i32, %arg1 : i32, %arg2 : i32):
      %7 = arith.muli %arg0, %arg1 : i32
      memref_stream.yield %7 : i32
    }
  }
}
