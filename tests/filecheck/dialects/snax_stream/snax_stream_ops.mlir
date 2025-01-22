// RUN: XDSL_ROUNDTRIP

"accfg.accelerator"() <{
    name = @accelerator_with_streamers,
    fields = {}, launch_fields = {}, barrier = 0
}> {
    "streamer_config" = #snax.streamer_config< r[temp=n-n, spat=n-n], r[temp=n-n, spat=n-n], w[temp=n-n, spat=n-n]>
} : () -> ()

%x, %y, %z = "test.op"() : () -> (index, index, index)

"snax_stream.streaming_region"(%x, %y, %z) <{
      "stride_patterns" = [
              #snax_stream.stride_pattern<ub = [16, 8], ts = [13, 7], ss = [8, 1]>, 
              #snax_stream.stride_pattern<ub = [19, 7], ts = [13, 7], ss = [8, 1]>, 
              #snax_stream.stride_pattern<ub = [13, 2], ts = [13, 7], ss = [8, 1]>
      ], "operandSegmentSizes" = array<i32: 2, 1>,
      "accelerator" = "accelerator_with_streamers"}> ({
^0(%3 : !dart.stream<i64>, %4 : !dart.stream<i64>, %5 : !dart.stream<i64>):
  %0 = "test.op" (%3, %4): (!dart.stream<i64>, !dart.stream<i64>) -> !dart.stream<i64>
  "dart.yield" (%0): (!dart.stream<i64>) -> ()
}) : (index, index, index) -> ()

//CHECK:      builtin.module {
//CHECK-NEXT:   "accfg.accelerator"() <{name = @accelerator_with_streamers, fields = {}, launch_fields = {}, barrier = 0 : i64}> {streamer_config = #snax.streamer_config<r[temp=n-n, spat=n-n], r[temp=n-n, spat=n-n], w[temp=n-n, spat=n-n]>} : () -> ()
//CHECK-NEXT:   %x, %y, %z = "test.op"() : () -> (index, index, index)
//CHECK-NEXT:   "snax_stream.streaming_region"(%x, %y, %z) <{stride_patterns = [#snax_stream.stride_pattern<ub = [16, 8], ts = [13, 7], ss = [8, 1]>, #snax_stream.stride_pattern<ub = [19, 7], ts = [13, 7], ss = [8, 1]>, #snax_stream.stride_pattern<ub = [13, 2], ts = [13, 7], ss = [8, 1]>], operandSegmentSizes = array<i32: 2, 1>, accelerator = "accelerator_with_streamers"}> ({
//CHECK-NEXT:   ^0(%0 : !dart.stream<i64>, %1 : !dart.stream<i64>, %2 : !dart.stream<i64>):
//CHECK-NEXT:     %3 = "test.op"(%0, %1) : (!dart.stream<i64>, !dart.stream<i64>) -> !dart.stream<i64>
//CHECK-NEXT:     dart.yield %3 : !dart.stream<i64>
//CHECK-NEXT:   }) : (index, index, index) -> ()
//CHECK-NEXT: }
