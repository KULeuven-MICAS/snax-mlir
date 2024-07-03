#streamer_add_attributes = {
  indexing_maps = [
    affine_map<(n) -> (n)>,
    affine_map<(n) -> (n)>,
    affine_map<(n) -> (n)>
  ],
  iterator_types = ["parallel"],
  library_call = "snax_alu" 
}

func.func private @snax_is_dm_core() -> i1
func.func private @snax_is_compute_core() -> i1
func.func public @streamer_add_tiled(%A: memref<128xi64, "L3">,
                             %B: memref<128xi64, "L3">,
                             %D: memref<128xi64, "L3">) -> () {
    %is_dm_core = func.call @snax_is_dm_core() : () -> i1
    %is_compute_core = func.call @snax_is_compute_core () : () -> i1
    %all_good = arith.constant 0 : i32
    // This code is run on both cores, note that only the DM core actually runs this, 
    // and that a barrier is called inside!
    %A_L1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<128xi64, "L1"> 
    %B_L1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<128xi64, "L1">
    %D_L1 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<128xi64, "L1">
    %tile_size = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    "snax.mcycle"() : () -> ()
    // Here goes all the code that is run on the DM core
    "scf.if"(%is_dm_core) ({
      "snax.mcycle"() : () -> ()
      // Perform the memory transfer on a subview of the memref
      scf.for %iv = %c0 to %c128 step %tile_size {
          "snax.mcycle"() : () -> ()
          %tiled_A = "memref.subview"(%A, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L3">, index, index) -> memref<?xi64, strided<[1], offset: ?> , "L3">
          %tiled_B = "memref.subview"(%B, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L3">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L3">
          %tiled_A_L1 = "memref.subview"(%A_L1, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
          %tiled_B_L1 = "memref.subview"(%B_L1, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
          "memref.copy"(%tiled_A, %tiled_A_L1) : (memref<?xi64, strided<[1], offset:?>, "L3">, memref<?xi64, strided<[1],offset: ?>, "L1">) -> ()
          "memref.copy"(%tiled_B, %tiled_B_L1) : (memref<?xi64, strided<[1], offset:?>, "L3">, memref<?xi64, strided<[1],offset: ?>, "L1">) -> ()
          scf.yield
      }
      // Synchronize with compute core
      "snax.cluster_sync_op"() : () -> ()
      // Wait for compute core to finish computing here
      "snax.cluster_sync_op"() : () -> ()
      // Send back output from L1 to L3
      scf.for %iv = %c0 to %c128 step %tile_size {
          "snax.mcycle"() : () -> ()
          %tiled_D = "memref.subview"(%D, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L3">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L3">
          %tiled_D_L1 = "memref.subview"(%D_L1, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
          "memref.copy"(%tiled_D_L1, %tiled_D) : (memref<?xi64, strided<[1], offset: ?>, "L1">, memref<?xi64, strided<[1], offset: ?>, "L3">) -> ()
          scf.yield
      }
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
      scf.for %iv = %c0 to %c128 step %tile_size {
          "snax.mcycle"() : () -> ()
          %tiled_A_L1 = "memref.subview"(%A_L1, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64:-9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
          %tiled_B_L1 = "memref.subview"(%B_L1, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64:-9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
          %tiled_D_L1 = "memref.subview"(%D_L1, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64:-9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
          "linalg.generic"(%tiled_A_L1, %tiled_B_L1, %tiled_D_L1) <{indexing_maps = [affine_map<(n) -> (n)>, affine_map<(n) -> (n)>, affine_map<(n) -> (n)>], iterator_types = [#linalg.iterator_type<parallel>], library_call = "snax_alu", operandSegmentSizes = array<i32: 2, 1>}> ({
          ^bb0(%arg3: i64, %arg4: i64, %arg5: i64):
            %10 = "arith.addi"(%arg3, %arg4) : (i64, i64) -> i64
            "linalg.yield"(%10) : (i64) -> ()
          }) : (memref<?xi64, strided<[1], offset: ?>, "L1">, memref<?xi64, strided<[1], offset: ?>, "L1">, memref<?xi64, strided<[1], offset: ?>, "L1">) -> ()
          // Synchronize with DM core
      }
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
