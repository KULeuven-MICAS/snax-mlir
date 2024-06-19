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
//func.func private @debug() -> ()
func.func public @simple_mult(%A: memref<64xi32, "L3">,
                             %B: memref<64xi32, "L3">,
                             %D: memref<64xi32, "L3">) -> () {
    %is_dm_core = func.call @snax_is_dm_core() : () -> i1
    %is_compute_core = func.call @snax_is_compute_core () : () -> i1
    %all_good = arith.constant 0 : i32
    // This code is run on both cores, note that only the DM core actually runs this, 
    // and that a barrier is called inside!
    %A_L1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64xi32, "L1"> 
    %B_L1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64xi32, "L1">
    %D_L1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64xi32, "L1">
    // Here goes all the code that is run on the DM core
    "scf.if"(%is_dm_core) ({
      %tile_size = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      // Perform the memory transfer on a subview of the memref
      scf.for %iv = %c0 to %c64 step %tile_size {
          %tiled_A = "memref.subview"(%A, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<64xi32, "L3">, index, index) -> memref<?xi32, strided<[1], offset: ?> , "L3">
          %tiled_B = "memref.subview"(%B, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<64xi32, "L3">, index, index) -> memref<?xi32, strided<[1], offset: ?>, "L3">
          %tiled_A_L1 = "memref.subview"(%A_L1, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<64xi32, "L1">, index, index) -> memref<?xi32, strided<[1], offset: ?>, "L1">
          %tiled_B_L1 = "memref.subview"(%B_L1, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<64xi32, "L1">, index, index) -> memref<?xi32, strided<[1], offset: ?>, "L1">
          "memref.copy"(%tiled_A, %tiled_A_L1) : (memref<?xi32, strided<[1], offset:?>, "L3">, memref<?xi32, strided<[1],offset: ?>, "L1">) -> ()
          "memref.copy"(%tiled_B, %tiled_B_L1) : (memref<?xi32, strided<[1], offset:?>, "L3">, memref<?xi32, strided<[1],offset: ?>, "L1">) -> ()
          scf.yield
      }
      // Synchronize with compute core
      "snax.cluster_sync_op"() : () -> ()
      // Wait for compute core to finish computing here
      "snax.cluster_sync_op"() : () -> ()
      // Send back output from L1 to L3
      %tiled_D = "memref.subview"(%D, %c64) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<64xi32, "L3">, index) -> memref<?xi32, strided<[1]>, "L3">
      %tiled_D_L1 = "memref.subview"(%D_L1, %c64) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<64xi32, "L1">, index) -> memref<?xi32, strided<[1]>, "L1">
      "memref.copy"(%tiled_D_L1, %tiled_D) : (memref<?xi32, strided<[1]>, "L1">, memref<?xi32, strided<[1]>, "L3">) -> ()
      "snax.cluster_sync_op"() : () -> ()
      scf.yield
    },{
      // Don't do anything if not a dm core
      scf.yield
    }) : (i1) -> ()
    // Here goes all the code that is run on the compute core
    "scf.if"(%is_compute_core) ({
      %tile_size = arith.constant 64 : index
      // Wait for input to come from DM core
      "snax.cluster_sync_op"() : () -> ()
      %tiled_A_L1 = "memref.subview"(%A_L1, %tile_size) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<64xi32, "L1">, index) -> memref<?xi32, strided<[1]>, "L1">
      %tiled_B_L1 = "memref.subview"(%B_L1, %tile_size) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<64xi32, "L1">, index) -> memref<?xi32, strided<[1]>, "L1">
      %tiled_D_L1 = "memref.subview"(%D_L1, %tile_size) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<64xi32, "L1">, index) -> memref<?xi32, strided<[1]>, "L1">
      "linalg.generic"(%tiled_A_L1, %tiled_B_L1, %tiled_D_L1) <{indexing_maps = [affine_map<(n) -> (n)>, affine_map<(n) -> (n)>, affine_map<(n) -> (n)>], iterator_types = [#linalg.iterator_type<parallel>], library_call = "snax_hwpe_mult", operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
        %10 = "arith.muli"(%arg3, %arg4) : (i32, i32) -> i32
        "linalg.yield"(%10) : (i32) -> ()
      }) : (memref<?xi32, strided<[1]>, "L1">, memref<?xi32, strided<[1]>, "L1">, memref<?xi32, strided<[1]>, "L1">) -> ()
      // Synchronize with DM core
      "snax.cluster_sync_op"() : () -> ()
      // Wait for output to go from L1 to L3
      "snax.cluster_sync_op"() : () -> ()
      //func.call @debug() : () -> ()
      scf.yield
    }, {
      scf.yield
      // Don't do anything if not a compute core
    }) : (i1) -> ()
    // return to main routine
    return
}
