// RUN: ./compiler/snax-opt --split-input-file %s -p guarded-linalg-to-memref-stream | filecheck %s

%0, %1, %2 = "test.op"() : () -> (memref<16xi64>, memref<16xi64>, memref<16xi64>)
linalg.generic {library_call="snax_alu", indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<16xi64>, memref<16xi64>) outs(%2 : memref<16xi64>) {
^bb0(%in: i64, %in_0: i64, %out: i64):
  %3 = arith.addi %in, %in_0 : i64
  linalg.yield %3 : i64
}

// CHECK: linalg.generic
// CHECK: library_call = "snax_alu"

// -----

%0, %1, %2 = "test.op"() : () -> (memref<16xi64>, memref<16xi64>, memref<16xi64>)
linalg.generic {library_call="snax_alu_stream", indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<16xi64>, memref<16xi64>) outs(%2 : memref<16xi64>) {
^bb0(%in: i64, %in_0: i64, %out: i64):
  %3 = arith.addi %in, %in_0 : i64
  linalg.yield %3 : i64
}

// CHECK: memref_stream.generic
// CHECK: library_call = "snax_alu"
