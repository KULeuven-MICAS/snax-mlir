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
//CHECK-NEXT:   "snax_stream.streaming_region"(%0, %1, %2) <{"stride_patterns" = [#snax_stream.stride_pattern<ub = [#builtin.int<4>], ts = [32], ss = [8]>, #snax_stream.stride_pattern<ub = [#builtin.int<4>], ts = [32], ss = [8]>, #snax_stream.stride_pattern<ub = [#builtin.int<4>], ts = [32], ss = [8]>], "accelerator" = "snax_alu", "operandSegmentSizes" = array<i32: 2, 1>}> ({
//CHECK-NEXT:   ^0(%a : !stream.readable<i64>, %b : !stream.readable<i64>, %c : !stream.writable<i64>):
//CHECK-NEXT:     "test.op"(%a, %b, %c) : (!stream.readable<i64>, !stream.readable<i64>, !stream.writable<i64>) -> ()
//CHECK-NEXT:   }) : (index, index, index) -> ()
