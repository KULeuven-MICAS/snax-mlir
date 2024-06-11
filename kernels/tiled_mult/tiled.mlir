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
func.func public @simple_mult(%A: memref<64xi32, 0 : i32>,
                             %B: memref<64xi32, 0: i32>,
                             %D: memref<64xi32, 0: i32>) -> () {
    %is_dm_core = func.call @snax_is_dm_core() : () -> i1
    %is_compute_core = func.call @snax_is_compute_core () : () -> i1
    %all_good = arith.constant 0 : i32
    %tile_size = arith.constant 4 : index
    // This code is run on both cores, note that only the DM core actually runs this, 
    // and that a barrier is called inside!
    %A_L1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64xi32, 1: i32>
    %B_L1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64xi32, 1: i32>
    %D_L1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64xi32, 1: i32>
    // Here goes all the code that is run on the DM core
    "scf.if"(%is_dm_core) ({
      // Perform the memory transfer on a subview of the memref
      %tiled_A = memref.subview %A  [0][64][1] : memref<64xi32, 0: i32> to memref<64xi32, strided<[1]>, 0: i32>
      %tiled_B = memref.subview %B  [0][64][1] : memref<64xi32, 0: i32> to memref<64xi32, strided<[1]>, 0: i32>
      %tiled_D = memref.subview %D  [0][64][1] : memref<64xi32, 0: i32> to memref<64xi32, strided<[1]>, 0: i32>
      %tiled_A_L1 = memref.subview %A_L1 [0][64][1] : memref<64xi32, 1: i32> to memref<64xi32, strided<[1]>, 1: i32>
      %tiled_B_L1 = memref.subview %B_L1 [0][64][1] : memref<64xi32, 1: i32> to memref<64xi32, strided<[1]>, 1: i32>
      %tiled_D_L1 = memref.subview %D_L1 [0][64][1] : memref<64xi32, 1: i32> to memref<64xi32, strided<[1]>, 1: i32>
      "memref.copy"(%tiled_A, %tiled_A_L1) : (memref<64xi32, strided<[1]>, 0: i32>, memref<64xi32, strided<[1]>, 1 : i32>) -> ()
      "memref.copy"(%tiled_B, %tiled_B_L1) : (memref<64xi32, strided<[1]>, 0: i32>, memref<64xi32, strided<[1]>, 1 : i32>) -> ()
      // Synchronize with compute core
      "snax.cluster_sync_op"() : () -> ()
      // Wait for compute core to finish computing here
      "snax.cluster_sync_op"() : () -> ()
      // Send back output from L1 to L3
      "memref.copy"(%tiled_D_L1, %tiled_D) : (memref<64xi32, strided<[1]>, 1: i32>, memref<64xi32, strided<[1]>, 0 : i32>) -> ()
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
      %tiled_A_L1 = memref.subview %A_L1 [0][64][1] : memref<64xi32, 1: i32> to memref<64xi32, strided<[1]>, 1: i32>
      %tiled_B_L1 = memref.subview %B_L1 [0][64][1] : memref<64xi32, 1: i32> to memref<64xi32, strided<[1]>, 1: i32>
      %tiled_D_L1 = memref.subview %D_L1 [0][64][1] : memref<64xi32, 1: i32> to memref<64xi32, strided<[1]>, 1: i32>
      linalg.generic #simple_mult_attributes
      ins(%tiled_A_L1, %tiled_B_L1: memref<64xi32, strided<[1]>, 1: i32>, memref<64xi32, strided<[1]>, 1 : i32>)
      outs(%tiled_D_L1: memref<64xi32, strided<[1]>, 1: i32>) {
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
