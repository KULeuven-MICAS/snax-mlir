#simple_mult_attributes = {
  indexing_maps = [
    affine_map<(n) -> (n)>,
    affine_map<(n) -> (n)>,
    affine_map<(n) -> (n)>
  ],
  iterator_types = ["parallel"],
  library_call = "snax_hwpe_mult" 
}

// https://github.com/pulp-platform/hwpe-mac-engine
// in 'simple_mult' mode, it takes two 32bit fixed-point streams (vectors),
// A, B and computes D = A * B where '*' is the elementwise product.
func.func private @snax_is_dm_core() -> i1
func.func private @snax_is_compute_core() -> i1
func.func public @simple_mult(%A: memref<128xi32, "L3">,
                             %B: memref<128xi32, "L3">,
                             %D: memref<128xi32, "L3">) -> () {
    %is_dm_core = func.call @snax_is_dm_core() : () -> i1
    %is_compute_core = func.call @snax_is_compute_core () : () -> i1
    %all_good = arith.constant 0 : i32
    // This code is run on both cores, note that only the DM core actually runs this, 
    // and that a barrier is called inside!
    %A_L1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<128xi32, "L1">
    %B_L1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<128xi32, "L1">
    %D_L1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<128xi32, "L1">
    // Here goes all the code that is run on the DM core
    "scf.if"(%is_dm_core) ({
      // Copy input from L3 to L1
      "memref.copy"(%A, %A_L1) : (memref<128xi32, "L3">, memref<128xi32, "L1">) -> ()
      "memref.copy"(%B, %B_L1) : (memref<128xi32, "L3">, memref<128xi32, "L1">) -> ()
      // Synchronize with compute core
      "snax.cluster_sync_op"() : () -> ()
      // Wait for compute core to finish computing here
      "snax.cluster_sync_op"() : () -> ()
      // Send back output from L1 to L3
      "memref.copy"(%D_L1, %D) : (memref<128xi32, "L1">, memref<128xi32, "L3">) -> ()
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
      linalg.generic #simple_mult_attributes
      ins(%A_L1, %B_L1: memref<128xi32, "L1">, memref<128xi32, "L1">)
      outs(%D_L1: memref<128xi32, "L1">) {
      ^bb0(%a: i32, %b: i32, %d: i32):
        %r0 = arith.muli %a, %b : i32
        linalg.yield %r0 : i32
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
