// RUN: ./compiler/snax-opt --split-input-file %s -p schedule-memref-linalg | filecheck %s

builtin.module {
  func.func public @streamer_add(%arg0 : memref<16xi64>, %arg1 : memref<16xi64>, %arg2 : memref<16xi64>) {
    memref_stream.generic {
      bounds = [16],
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"],
      library_call = "snax_alu"
    } ins(%arg0, %arg1 : memref<16xi64>, memref<16xi64>) outs(%arg2 : memref<16xi64>) {
    ^0(%arg3 : i64, %arg4 : i64, %arg5 : i64):
      %0 = arith.addi %arg3, %arg4 : i64
      memref_stream.yield %0 : i64
    }
    func.return
  }
}

// CHECK:      indexing_maps = [
// CHECK-NEXT:   affine_map<(d0, d1) -> (((d0 * 4) + d1))>,
// CHECK-NEXT:   affine_map<(d0, d1) -> (((d0 * 4) + d1))>,
// CHECK-NEXT:   affine_map<(d0, d1) -> (((d0 * 4) + d1))>
// CHECK-NEXT: ],
