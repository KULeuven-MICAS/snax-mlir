// RUN: ./compiler/snax-opt --split-input-file %s -p insert-accfg-op{accelerator=snax_alu},insert-accfg-op{accelerator=snax_gemmx},dart-scheduler,dart-layout-resolution,convert-dart-to-snax-stream | filecheck %s

func.func public @streamer_add(%arg0 : memref<16xi64>, %arg1 : memref<16xi64>, %arg2 : memref<16xi64>) {
  "dart.operation"(%arg0, %arg1, %arg2) <{patterns = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], accelerator = "snax_alu", operandSegmentSizes = array<i32: 2, 1>}> ({
  ^0(%0 : !dart.stream<i64>, %1 : !dart.stream<i64>, %2 : !dart.stream<i64>):
    %3 = "dart.generic"(%0, %1) <{library_call = "snax_alu"}> ({
    ^1(%arg3 : i64, %arg4 : i64, %arg5 : i64):
      %4 = kernel.add %arg3, %arg4 : i64, i64 -> i64
      dart.yield %4 : i64
    }) : (!dart.stream<i64>, !dart.stream<i64>) -> !dart.stream<i64>
    dart.yield %3 : !dart.stream<i64>
  }) : (memref<16xi64>, memref<16xi64>, memref<16xi64>) -> ()
  func.return
}

// CHECK:       %0 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<16xi64>) -> index
// CHECK-NEXT:  %1 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<16xi64>) -> index
// CHECK-NEXT:  %2 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<16xi64>) -> index
// CHECK-NEXT:  "snax_stream.streaming_region"(%0, %1, %2) <{stride_patterns = [#snax_stream.stride_pattern<ub = [4], ts = [32], ss = [8]>, #snax_stream.stride_pattern<ub = [4], ts = [32], ss = [8]>, #snax_stream.stride_pattern<ub = [4], ts = [32], ss = [8]>], accelerator = "snax_alu", operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:  ^0(%3 : !dart.stream<i64>, %4 : !dart.stream<i64>, %5 : !dart.stream<i64>):
// CHECK-NEXT:    %6 = "dart.generic"(%3, %4) <{library_call = "snax_alu"}> ({
// CHECK-NEXT:    ^1(%arg3 : i64, %arg4 : i64, %arg5 : i64):
// CHECK-NEXT:      %7 = kernel.add %arg3, %arg4 : i64, i64 -> i64
// CHECK-NEXT:      dart.yield %7 : i64
// CHECK-NEXT:    }) : (!dart.stream<i64>, !dart.stream<i64>) -> !dart.stream<i64>
// CHECK-NEXT:    dart.yield %6 : !dart.stream<i64>
// CHECK-NEXT:  }) : (index, index, index) -> ()

// -----

func.func @streamer_matmul(%arg0 : memref<16x16xi8>, %arg1 : memref<16x16xi8, strided<[1, 16]>>, %arg2 : memref<16x16xi32>) {
  %0 = arith.constant 0 : i32
  "dart.operation"(%arg0, %arg1, %arg2) <{patterns = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], accelerator = "snax_gemmx", operandSegmentSizes = array<i32: 2, 1>}> ({
  ^0(%1 : !dart.stream<i8>, %2 : !dart.stream<i8>, %3 : !dart.stream<i32>):
    %4 = "dart.generic"(%1, %2, %0, %0) <{library_call = "snax_gemmx"}> ({
    ^1(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
      %5 = kernel.qmac %arg3, %arg4 zp_lhs : %arg5 zp_rhs : %arg6 : i8, i8, i32, i32 -> i32
      dart.yield %5 : i32
    }) : (!dart.stream<i8>, !dart.stream<i8>, i32, i32) -> !dart.stream<i32>
    dart.yield %4 : !dart.stream<i32>
  }) : (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32>) -> ()
  func.return
}

// CHECK:       "snax_stream.streaming_region"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{stride_patterns = [#snax_stream.stride_pattern<ub = [2, 2, 2], ts = [8, 128, 0], ss = [8]>, #snax_stream.stride_pattern<ub = [2, 2, 2], ts = [8, 0, 128], ss = [8]>, #snax_stream.stride_pattern<ub = [0, 0, 0], ts = [0, 0, 0], ss = [0]>, #snax_stream.stride_pattern<ub = [2, 2, 2], ts = [0, 0, 0], ss = [8, 64]>, #snax_stream.stride_pattern<ub = [2, 2, 2], ts = [0, 512, 32], ss = [8, 64]>], accelerator = "snax_gemmx", operandSegmentSizes = array<i32: 4, 1>}> ({
// CHECK-NEXT:  ^0(%{{.*}} : !dart.stream<i8>, %{{.*}} : !dart.stream<i8>, %{{.*}} : !dart.stream<i32>):
// CHECK-NEXT:    %{{.*}} = "dart.generic"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{library_call = "snax_gemmx"}> ({
// CHECK-NEXT:    ^1(%{{.*}} : i8, %{{.*}} : i8, %{{.*}} : i32, %{{.*}} : i32, %{{.*}} : i32):
// CHECK-NEXT:      %{{.*}} = kernel.qmac %{{.*}}, %{{.*}} zp_lhs : %{{.*}} zp_rhs : %{{.*}} : i8, i8, i32, i32 -> i32
// CHECK-NEXT:      dart.yield %{{.*}} : i32
// CHECK-NEXT:    }) : (!dart.stream<i8>, !dart.stream<i8>, i32, i32) -> !dart.stream<i32>
// CHECK-NEXT:    dart.yield %{{.*}} : !dart.stream<i32>
// CHECK-NEXT:  }) : (index, index, index, index, index) -> ()
