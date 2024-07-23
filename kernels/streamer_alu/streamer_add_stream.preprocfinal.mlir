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

    memref_stream.streaming_region {
      patterns = [
          #memref_stream.stride_pattern<ub = [4, 4], index_map = (d0, d1) -> (4 * d0 + d1)>,
          #memref_stream.stride_pattern<ub = [4, 4], index_map = (d0, d1) -> (4 * d0 + d1)>,
          #memref_stream.stride_pattern<ub = [4, 4], index_map = (d0, d1) -> (4 * d0 + d1)>
      ]
    } ins(%4, %7 : memref<?xi64, "L1">, memref<?xi64, "L1">) outs(%10 : memref<?xi64, "L1">) attrs = {accelerator="snax_alu"} {
    ^bb0(%103: !stream.readable<i64>, %104: !stream.readable<i64>, %105: !stream.writable<i64>):

      // body of this streaming region does not really matter, as it is not 
      // yet considered by the transformation passes

      %100 = stream.read from %103 : i64
      %101 = stream.read from %104 : i64
      %102 = arith.addi %100, %101 : i64
      stream.write %102 to %105 : i64

    }

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

func.func private @snax_is_compute_core() -> i1
func.func private @snax_is_dm_core() -> i1
