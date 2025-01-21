// RUN: ./compiler/snax-opt --split-input-file %s -p insert-accfg-op{accelerator=snax_alu},insert-accfg-op{accelerator=snax_gemmx},convert-stream-to-snax-stream | filecheck %s

func.func public @streamer_add(%arg0 : memref<16xi64>, %arg1 : memref<16xi64>, %arg2 : memref<16xi64>) {
  "stream.streaming_region"(%arg0, %arg1, %arg2) <{patterns = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], accelerator = "snax_alu", operandSegmentSizes = array<i32: 2, 1>}> ({
  ^0(%0 : !stream.stream<i64>, %1 : !stream.stream<i64>, %2 : !stream.stream<i64>):
    %3 = "stream.generic"(%0, %1) <{library_call = "snax_alu"}> ({
    ^1(%arg3 : i64, %arg4 : i64, %arg5 : i64):
      %4 = kernel.add %arg3, %arg4 : i64, i64 -> i64
      stream.yield %4 : i64
    }) : (!stream.stream<i64>, !stream.stream<i64>) -> !stream.stream<i64>
    stream.yield %3 : !stream.stream<i64>
  }) : (memref<16xi64>, memref<16xi64>, memref<16xi64>) -> ()
  func.return
}

// CHECK:       %0 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<16xi64>) -> index
// CHECK-NEXT:  %1 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<16xi64>) -> index
// CHECK-NEXT:  %2 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<16xi64>) -> index
// CHECK-NEXT:  "snax_stream.streaming_region"(%0, %1, %2) <{stride_patterns = [#snax_stream.stride_pattern<ub = [4], ts = [32], ss = [8]>, #snax_stream.stride_pattern<ub = [4], ts = [32], ss = [8]>, #snax_stream.stride_pattern<ub = [4], ts = [32], ss = [8]>], accelerator = "snax_alu", operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:  ^0(%3 : !stream.stream<i64>, %4 : !stream.stream<i64>, %5 : !stream.stream<i64>):
// CHECK-NEXT:    %6 = "stream.generic"(%3, %4) <{library_call = "snax_alu"}> ({
// CHECK-NEXT:    ^1(%arg3 : i64, %arg4 : i64, %arg5 : i64):
// CHECK-NEXT:      %7 = kernel.add %arg3, %arg4 : i64, i64 -> i64
// CHECK-NEXT:      stream.yield %7 : i64
// CHECK-NEXT:    }) : (!stream.stream<i64>, !stream.stream<i64>) -> !stream.stream<i64>
// CHECK-NEXT:    stream.yield %6 : !stream.stream<i64>
// CHECK-NEXT:  }) : (index, index, index) -> ()

// -----

func.func @streamer_matmul(%arg0 : memref<16x16xi8>, %arg1 : memref<16x16xi8, strided<[1, 16]>>, %arg2 : memref<16x16xi32>) {
  %0 = arith.constant 0 : i32
  "stream.streaming_region"(%arg0, %arg1, %arg2) <{patterns = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], accelerator = "snax_gemmx", operandSegmentSizes = array<i32: 2, 1>}> ({
  ^0(%1 : !stream.stream<i8>, %2 : !stream.stream<i8>, %3 : !stream.stream<i32>):
    %4 = "stream.generic"(%1, %2, %0, %0) <{library_call = "snax_gemmx"}> ({
    ^1(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
      %5 = kernel.qmac %arg3, %arg4 zp_lhs : %arg5 zp_rhs : %arg6 : i8, i8, i32, i32 -> i32
      stream.yield %5 : i32
    }) : (!stream.stream<i8>, !stream.stream<i8>, i32, i32) -> !stream.stream<i32>
    stream.yield %4 : !stream.stream<i32>
  }) : (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32>) -> ()
  func.return
}

// CHECK:       "snax_stream.streaming_region"(%1, %2, %3, %4, %5) <{stride_patterns = [#snax_stream.stride_pattern<ub = [2, 2, 2], ts = [8, 0, 128], ss = [8]>, #snax_stream.stride_pattern<ub = [2, 2, 2], ts = [8, 128, 0], ss = [8]>, #snax_stream.stride_pattern<ub = [0, 0, 0], ts = [0, 0, 0], ss = [0]>, #snax_stream.stride_pattern<ub = [2, 2, 2], ts = [0, 0, 0], ss = [8, 64]>, #snax_stream.stride_pattern<ub = [2, 2, 2], ts = [0, 32, 512], ss = [8, 64]>], accelerator = "snax_gemmx", operandSegmentSizes = array<i32: 4, 1>}> ({
// CHECK-NEXT:  ^0(%6 : !stream.stream<i8>, %7 : !stream.stream<i8>, %8 : !stream.stream<i32>):
// CHECK-NEXT:    %9 = "stream.generic"(%6, %7, %0, %0) <{library_call = "snax_gemmx"}> ({
// CHECK-NEXT:    ^1(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
// CHECK-NEXT:      %10 = kernel.qmac %arg3, %arg4 zp_lhs : %arg5 zp_rhs : %arg6 : i8, i8, i32, i32 -> i32
// CHECK-NEXT:      stream.yield %10 : i32
// CHECK-NEXT:    }) : (!stream.stream<i8>, !stream.stream<i8>, i32, i32) -> !stream.stream<i32>
// CHECK-NEXT:    stream.yield %9 : !stream.stream<i32>
// CHECK-NEXT:  }) : (index, index, index, index, index) -> ()
