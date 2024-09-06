// RUN: ./compiler/snax-opt --split-input-file %s -p insert-accfg-op{accelerator=snax_alu},stream-snaxify | filecheck %s

%A, %B, %C = "test.op"() : () -> (memref<16xi64>, memref<16xi64>, memref<16xi64>)

memref_stream.streaming_region {
    patterns = [
        #memref_stream.stride_pattern<ub = [4, 4], index_map = (d0, d1) -> (4 * d0 + d1)>,
        #memref_stream.stride_pattern<ub = [4, 4], index_map = (d0, d1) -> (4 * d0 + d1)>,
        #memref_stream.stride_pattern<ub = [4, 4], index_map = (d0, d1) -> (4 * d0 + d1)>
    ]
} ins(%A, %B : memref<16xi64>, memref<16xi64>) outs(%C : memref<16xi64>) attrs = {accelerator="snax_alu"} {
^bb0(%a: !stream.readable<i64>, %b: !stream.readable<i64>, %c: !stream.writable<i64>):
    "test.op"(%a, %b, %c) : (!stream.readable<i64>, !stream.readable<i64>, !stream.writable<i64>) -> ()
}

//CHECK:        %0 = "memref.extract_aligned_pointer_as_index"(%A) : (memref<16xi64>) -> index
//CHECK-NEXT:   %1 = "memref.extract_aligned_pointer_as_index"(%B) : (memref<16xi64>) -> index
//CHECK-NEXT:   %2 = "memref.extract_aligned_pointer_as_index"(%C) : (memref<16xi64>) -> index
//CHECK-NEXT:   "snax_stream.streaming_region"(%0, %1, %2) <{"stride_patterns" = [#snax_stream.stride_pattern<ub = [4], ts = [32], ss = [8]>, #snax_stream.stride_pattern<ub = [4], ts = [32], ss = [8]>, #snax_stream.stride_pattern<ub = [4], ts = [32], ss = [8]>], "accelerator" = "snax_alu", "operandSegmentSizes" = array<i32: 2, 1>}> ({
//CHECK-NEXT:   ^0(%a : !stream.readable<i64>, %b : !stream.readable<i64>, %c : !stream.writable<i64>):
//CHECK-NEXT:     "test.op"(%a, %b, %c) : (!stream.readable<i64>, !stream.readable<i64>, !stream.writable<i64>) -> ()
//CHECK-NEXT:   }) : (index, index, index) -> ()

// -----


%0, %1, %2 = "test.op"() : () -> (memref<64xi32>, memref<64xi32>, memref<64xi32>)
memref_stream.streaming_region {
  patterns = [
    #memref_stream.stride_pattern<ub = [64], index_map = (d0) -> (d0)>,
    #memref_stream.stride_pattern<ub = [64], index_map = (d0) -> (d0)>,
    #memref_stream.stride_pattern<ub = [64], index_map = (d0) -> (d0)>
  ]
} ins(%0, %1 : memref<64xi32>, memref<64xi32>) outs(%2 : memref<64xi32>) {
^0(%3 : !stream.readable<i32>, %4 : !stream.readable<i32>, %5 : !stream.writable<i32>):
  memref_stream.generic {
    bounds = [64],
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"],
    library_call = "snax_alu"
  } ins(%3, %4 : !stream.readable<i32>, !stream.readable<i32>) outs(%5 : !stream.writable<i32>) {
  ^1(%arg0 : i32, %arg1 : i32, %arg2 : i32):
    %6 = arith.muli %arg0, %arg1 : i32
    memref_stream.yield %6 : i32
  }
}

// CHECK: snax_stream.streaming_region
// CHECK-SAME: "accelerator" = "snax_alu"

// -----

%A, %B, %C = "test.op"() : () -> (memref<16xi64, #tsl.tsl<[4, 4] -> (1, 5)>>, memref<16xi64, #tsl.tsl<[4, 4] -> (13, 39)>>, memref<16xi64, #tsl.tsl<[4, 4] -> (23, 3)>>)

memref_stream.streaming_region {
    patterns = [
        #memref_stream.stride_pattern<ub = [4, 4], index_map = (d0, d1) -> (4 * d0 + d1)>,
        #memref_stream.stride_pattern<ub = [4, 4], index_map = (d0, d1) -> (4 * d0 + d1)>,
        #memref_stream.stride_pattern<ub = [4, 4], index_map = (d0, d1) -> (4 * d0 + d1)>
    ]
} ins(%A, %B : memref<16xi64, #tsl.tsl<[4, 4] -> (1, 5)>>, memref<16xi64, #tsl.tsl<[4, 4] -> (13, 39)>>) outs(%C:  memref<16xi64, #tsl.tsl<[4, 4] -> (23, 3)>>) attrs = {accelerator="snax_alu"} {
^bb0(%a: !stream.readable<i64>, %b: !stream.readable<i64>, %c: !stream.writable<i64>):
    "test.op"(%a, %b, %c) : (!stream.readable<i64>, !stream.readable<i64>, !stream.writable<i64>) -> ()
}

// CHECK: "stride_patterns" = [#snax_stream.stride_pattern<ub = [4], ts = [8], ss = [40]>, #snax_stream.stride_pattern<ub = [4], ts = [104], ss = [312]>, #snax_stream.stride_pattern<ub = [4], ts = [184], ss = [24]>]
