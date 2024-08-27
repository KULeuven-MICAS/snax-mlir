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

//CHECK:      patterns = [
//CHECK-NEXT:   #memref_stream.stride_pattern<ub = [4, 4], index_map = (d0, d1) -> (((d0 * 4) + d1))>,
//CHECK-NEXT:   #memref_stream.stride_pattern<ub = [4, 4], index_map = (d0, d1) -> (((d0 * 4) + d1))>,
//CHECK-NEXT:   #memref_stream.stride_pattern<ub = [4, 4], index_map = (d0, d1) -> (((d0 * 4) + d1))>
//CHECK-NEXT: ]

// -----

builtin.module {
  func.func @streamer_matmul(%arg0 : memref<16x16xi8>, %arg1 : memref<16x16xi8, strided<[1, 16]>>, %arg2 : memref<16x16xi32>) {
    %0 = arith.constant 0 : i32
    memref_stream.generic {
      bounds = [16, 16, 16],
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> ()>,
        affine_map<(d0, d1, d2) -> ()>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      library_call = "snax_gemm"
    } ins(%arg0, %arg1, %0, %0 : memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, i32, i32) outs(%arg2 : memref<16x16xi32>) {
    ^0(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
      %1 = arith.extsi %arg3 : i8 to i32
      %2 = arith.subi %1, %arg5 : i32
      %3 = arith.extsi %arg4 : i8 to i32
      %4 = arith.subi %3, %arg6 : i32
      %5 = arith.muli %2, %4 : i32
      %6 = arith.addi %arg7, %5 : i32
      memref_stream.yield %6 : i32
    }
    func.return
  }
}

//CHECK:      patterns = [
//CHECK-NEXT:   #memref_stream.stride_pattern<ub = [2, 2, 2, 8, 8, 8], index_map = (d0, d1, d2, d3, d4, d5) -> (((d0 * 8) + d3), ((d2 * 8) + d5))>,
//CHECK-NEXT:   #memref_stream.stride_pattern<ub = [2, 2, 2, 8, 8, 8], index_map = (d0, d1, d2, d3, d4, d5) -> (((d2 * 8) + d5), ((d1 * 8) + d4))>,
//CHECK-NEXT:   #memref_stream.stride_pattern<ub = [2, 2, 2, 8, 8, 8], index_map = (d0, d1, d2, d3, d4, d5) -> (((d0 * 8) + d3), ((d1 * 8) + d4))>
//CHECK-NEXT: ]
