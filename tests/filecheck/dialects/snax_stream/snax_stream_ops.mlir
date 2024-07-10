// RUN: XDSL_ROUNDTRIP

%x, %y, %z = "test.op"() : () -> (index, index, index)

"snax_stream.streaming_region"(%x, %y, %z) <{
      "stride_pattern" = [
              #snax_stream.stride_pattern<ub = [16, 8], ts = [13, 7], ss = [8, 1]>, 
              #snax_stream.stride_pattern<ub = [19, 7], ts = [13, 7], ss = [8, 1]>, 
              #snax_stream.stride_pattern<ub = [13, 2], ts = [13, 7], ss = [8, 1]>
      ], "operandSegmentSizes" = array<i32: 2, 1>}> ({
^0(%3 : !stream.readable<i64>, %4 : !stream.readable<i64>, %5 : !stream.writable<i64>):
  %0 = stream.read from %3 : i64
  %1 = stream.read from %4 : i64
  %2 = arith.addi %0, %1 : i64
  stream.write %2 to %5 : i64
}) : (index, index, index) -> ()

//CHECK:      builtin.module {
//CHECK-NEXT:   %x, %y, %z = "test.op"() : () -> (index, index, index)
//CHECK-NEXT:   "snax_stream.streaming_region"(%x, %y, %z) <{"stride_pattern" = [#snax_stream.stride_pattern<ub = [16, 8], ts = [13, 7], ss = [8, 1]>, #snax_stream.stride_pattern<ub = [19, 7], ts = [13, 7], ss = [8, 1]>, #snax_stream.stride_pattern<ub = [13, 2], ts = [13, 7], ss = [8, 1]>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
//CHECK-NEXT:   ^0(%0 : !stream.readable<i64>, %1 : !stream.readable<i64>, %2 : !stream.writable<i64>):
//CHECK-NEXT:     %3 = stream.read from %0 : i64
//CHECK-NEXT:     %4 = stream.read from %1 : i64
//CHECK-NEXT:     %5 = arith.addi %3, %4 : i64
//CHECK-NEXT:     stream.write %5 to %2 : i64
//CHECK-NEXT:   }) : (index, index, index) -> ()
//CHECK-NEXT: }

