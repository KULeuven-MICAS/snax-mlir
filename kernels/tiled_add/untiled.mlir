#streamer_add_attributes = {
  indexing_maps = [
    affine_map<(n) -> (n)>,
    affine_map<(n) -> (n)>,
    affine_map<(n) -> (n)>
  ],
  iterator_types = ["parallel"],
  library_call = "snax_alu"
}

func.func private @snax_global_core_idx() -> i32
func.func public @streamer_add_tiled(%A: memref<128xi64, "L3">,
                             %B: memref<128xi64, "L3">,
                             %D: memref<128xi64, "L3">) -> () {
    %compute_core_constant = arith.constant 0 : i32
    %dm_core_constant = arith.constant 1 : i32
    %which_core_id = func.call @snax_global_core_idx() : () ->  i32
    %is_compute_core = arith.cmpi eq, %which_core_id, %compute_core_constant : i32
    %is_dm_core = arith.cmpi eq, %which_core_id, %dm_core_constant : i32
    %all_good = arith.constant 0 : i32
    // This code is run on both cores, note that only the DM core actually runs this, 
    // and that a barrier is called inside!
    %A_L1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<128xi64, "L1">
    %B_L1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<128xi64, "L1">
    %D_L1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<128xi64, "L1">
    // Here goes all the code that is run on the DM core
    "scf.if"(%is_dm_core) ({
      // Copy input from L3 to L1
      "memref.copy"(%A, %A_L1) : (memref<128xi64, "L3">, memref<128xi64, "L1">) -> ()
      "memref.copy"(%B, %B_L1) : (memref<128xi64, "L3">, memref<128xi64, "L1">) -> ()
      // Synchronize with compute core
      "snax.cluster_sync_op"() : () -> ()
      // Wait for compute core to finish computing here
      "snax.cluster_sync_op"() : () -> ()
      // Send back output from L1 to L3
      "memref.copy"(%D_L1, %D) : (memref<128xi64, "L1">, memref<128xi64, "L3">) -> ()
      "snax.cluster_sync_op"() : () -> ()
      scf.yield
    },{
      // Don't do anything if not a dm core
      scf.yield
    }) : (i1) -> ()
    // Here goes all the code that is run on the compute core
    "scf.if"(%is_compute_core) ({
      // Wait for input to come from DM core
      "snax.cluster_sync_op"() : () -> ()
      linalg.generic #streamer_add_attributes
      ins(%A_L1, %B_L1: memref<128xi64, "L1">, memref<128xi64, "L1">)
      outs(%D_L1: memref<128xi64, "L1">) {
      ^bb0(%a: i64, %b: i64, %d: i64):
        %r0 = arith.addi %a, %b : i64
        linalg.yield %r0 : i64
      }
      // Synchronize with DM core
      "snax.cluster_sync_op"() : () -> ()
      // Wait for output to go from L1 to L3
      "snax.cluster_sync_op"() : () -> ()
      scf.yield
    }, {
      scf.yield
      // Don't do anything if not a compute core
    }) : (i1) -> ()
    // return to main routine
    return
}
