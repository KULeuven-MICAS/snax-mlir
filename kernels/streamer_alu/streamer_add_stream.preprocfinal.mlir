func.func public @streamer_add(%arg0 : memref<?xi64, "L3">, %arg1 : memref<?xi64, "L3">, %arg2 : memref<?xi64, "L3">) {
  %0 = func.call @snax_is_compute_core() : () -> i1
  %1 = func.call @snax_is_dm_core() : () -> i1
  %2 = arith.constant 0 : index
  %3 = "memref.dim"(%arg0, %2) : (memref<?xi64, "L3">, index) -> index
  %4 = memref.alloc(%3) {"alignment" = 64 : i64} : memref<?xi64, "L1">
  %5 = arith.constant 0 : index
  %6 = "memref.dim"(%arg1, %5) : (memref<?xi64, "L3">, index) -> index
  %7 = memref.alloc(%6) {"alignment" = 64 : i64} : memref<?xi64, "L1">
  %8 = arith.constant 0 : index
  %9 = "memref.dim"(%arg2, %8) : (memref<?xi64, "L3">, index) -> index
  %10 = memref.alloc(%9) {"alignment" = 64 : i64} : memref<?xi64, "L1">
  "scf.if"(%1) ({
    "memref.copy"(%arg1, %7) : (memref<?xi64, "L3">, memref<?xi64, "L1">) -> ()
    "memref.copy"(%arg0, %4) : (memref<?xi64, "L3">, memref<?xi64, "L1">) -> ()
    scf.yield
  }, {
  }) : (i1) -> ()
  "snax.cluster_sync_op"() : () -> ()
  "scf.if"(%0) ({

    %x = "memref.extract_aligned_pointer_as_index" (%4) : (memref<?xi64, "L1">) -> (index)
    %y = "memref.extract_aligned_pointer_as_index" (%7) : (memref<?xi64, "L1">) -> (index)
    %z = "memref.extract_aligned_pointer_as_index" (%10) : (memref<?xi64, "L1">) -> (index)

    "snax_stream.streaming_region"(%x, %y, %z) <{
          "stride_pattern" = [
                  #snax_stream.stride_pattern<ub = [4], ts = [32], ss = [8]>, 
                  #snax_stream.stride_pattern<ub = [4], ts = [32], ss = [8]>, 
                  #snax_stream.stride_pattern<ub = [4], ts = [32], ss = [8]>
          ], "operandSegmentSizes" = array<i32: 2, 1>,
          "accelerator" = "snax_alu"}> ({
    ^0(%103 : !stream.readable<i64>, %104 : !stream.readable<i64>, %105 : !stream.writable<i64>):

      // body of this streaming region does not really matter, as it is not 
      // yet considered by the transformation passes

      %100 = stream.read from %103 : i64
      %101 = stream.read from %104 : i64
      %102 = arith.addi %100, %101 : i64
      stream.write %102 to %105 : i64

    }) : (index, index, index) -> ()


    scf.yield
  }, {
  }) : (i1) -> ()
  "snax.cluster_sync_op"() : () -> ()
  "scf.if"(%1) ({
    "memref.copy"(%10, %arg2) : (memref<?xi64, "L1">, memref<?xi64, "L3">) -> ()
    scf.yield
  }, {
  }) : (i1) -> ()
  func.return
}

"accfg.accelerator"() <{"name" = @snax_alu, 
  "fields" = {
    "loop_bound_0" = 960 : i32, 
    "a_tstride_0" = 961 : i32, 
    "b_tstride_0" = 962 : i32, 
    "c_tstride_0" = 963 : i32, 
    "a_sstride_0" = 964 : i32, 
    "b_sstride_0" = 965 : i32, 
    "c_sstride_0" = 966 : i32, 
    "a_ptr" = 967 : i32, 
    "b_ptr" = 968 : i32, 
    "c_ptr" = 969 : i32, 
    "alu_mode" = 972 : i32, 
    "loop_bound_alu" = 973 : i32}, 
  "launch_fields" = {
    "launch_streamer" = 970 : i32, 
    "launch_alu" = 974 : i32}, 
  "barrier" = 975 : i32}> {
    "streamer_config" = #snax.streamer_config<r[1, 1], r[1, 1], r[1, 1]>
} : () -> ()

func.func private @snax_is_compute_core() -> i1
func.func private @snax_is_dm_core() -> i1
