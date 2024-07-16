// RUN: XDSL_VERIFY_DIAG

"snax_stream.streaming_region"() <{
  "stride_pattern" = [],
  "operandSegmentSizes" = array<i32: 0, 0>,
  "accelerator" = "accelerator_with_streamers"}> ({
^0():
}) : () -> ()

// CHECK: Operation does not verify: AcceleratorOp not found!

// -----

"accfg.accelerator"() <{
    name = @accelerator_without_streamers,
    fields = {}, launch_fields = {}, barrier = 0
}> {} : () -> ()

"snax_stream.streaming_region"() <{
  "stride_pattern" = [],
  "operandSegmentSizes" = array<i32: 0, 0>,
  "accelerator" = "accelerator_without_streamers"}> ({
^0():
}) : () -> ()

// CHECK: Operation does not verify: Specified accelerator does not contain a StreamerConfigurationAttr

// -----

"accfg.accelerator"() <{
    name = @accelerator_with_streamers,
    fields = {}, launch_fields = {}, barrier = 0
}> {
    "streamer_config" = #snax.streamer_config< r[2,2], r[2,2], w[2,2]>
} : () -> ()

"snax_stream.streaming_region"() <{
  "stride_pattern" = [],
  "operandSegmentSizes" = array<i32: 0, 0>,
  "accelerator" = "accelerator_with_streamers"}> ({
^0():
}) : () -> ()

// CHECK: Operation does not verify: Number of streamers does not equal number of stride patterns

// -----

"accfg.accelerator"() <{
    name = @accelerator_with_streamers,
    fields = {}, launch_fields = {}, barrier = 0
}> {
    "streamer_config" = #snax.streamer_config< r[2,2], r[2,2], w[2,2]>
} : () -> ()

"snax_stream.streaming_region"() <{
  "stride_pattern" = [
              #snax_stream.stride_pattern<ub = [16, 8, 3], ts = [13, 7, 5], ss = [8, 1]>,
              #snax_stream.stride_pattern<ub = [19, 7], ts = [13, 7], ss = [8, 1]>,
              #snax_stream.stride_pattern<ub = [13, 2], ts = [13, 7], ss = [8, 1]>
  ], "operandSegmentSizes" = array<i32: 2, 1>,
  "accelerator" = "accelerator_with_streamers"}> ({
^0():
}) : () -> ()

// CHECK: Operation does not verify: Temporal stride pattern exceeds streamer dimensionality

// -----

"accfg.accelerator"() <{
    name = @accelerator_with_streamers,
    fields = {}, launch_fields = {}, barrier = 0
}> {
    "streamer_config" = #snax.streamer_config< r[2,2], r[2,2], w[2,2]>
} : () -> ()

"snax_stream.streaming_region"() <{
  "stride_pattern" = [
              #snax_stream.stride_pattern<ub = [16, 8], ts = [13, 7], ss = [9, 8, 1]>,
              #snax_stream.stride_pattern<ub = [19, 7], ts = [13, 7], ss = [8, 1]>,
              #snax_stream.stride_pattern<ub = [13, 2], ts = [13, 7], ss = [8, 1]>
  ], "operandSegmentSizes" = array<i32: 2, 1>,
  "accelerator" = "accelerator_with_streamers"}> ({
^0():
}) : () -> ()

// CHECK: Operation does not verify: Spatial stride pattern exceeds streamer dimensionality





