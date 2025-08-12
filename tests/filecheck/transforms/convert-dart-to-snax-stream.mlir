// RUN: snax-opt --split-input-file %s -p insert-accfg-op{accelerator=snax_alu},insert-accfg-op{accelerator=snax_gemmx},dart-scheduler,dart-layout-resolution,convert-dart-to-snax-stream | filecheck %s

func.func public @streamer_add(%arg0 : memref<16xi64>, %arg1 : memref<16xi64>, %arg2 : memref<16xi64>) {
  "dart.operation"(%arg0, %arg1, %arg2) <{patterns = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], accelerator = "snax_alu", operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%0 : !dart.stream<i64>, %1 : !dart.stream<i64>, %2 : !dart.stream<i64>):
    %3 = "dart.generic"(%0, %1) <{library_call = "snax_alu"}> ({
    ^bb1(%arg3 : i64, %arg4 : i64, %arg5 : i64):
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
// CHECK-NEXT:  ^bb0(%3 : !dart.stream<i64>, %4 : !dart.stream<i64>, %5 : !dart.stream<i64>):
// CHECK-NEXT:    %6 = "dart.generic"(%3, %4) <{library_call = "snax_alu"}> ({
// CHECK-NEXT:    ^bb1(%arg3 : i64, %arg4 : i64, %arg5 : i64):
// CHECK-NEXT:      %7 = kernel.add %arg3, %arg4 : i64, i64 -> i64
// CHECK-NEXT:      dart.yield %7 : i64
// CHECK-NEXT:    }) : (!dart.stream<i64>, !dart.stream<i64>) -> !dart.stream<i64>
// CHECK-NEXT:    dart.yield %6 : !dart.stream<i64>
// CHECK-NEXT:  }) : (index, index, index) -> ()
