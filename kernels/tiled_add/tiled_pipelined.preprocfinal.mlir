#simple_mult_attributes = {
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
    %c2 = arith.constant 2 : index
    // hot loop begins after two tiles
    %two_tile_size = arith.muli %c2, %tile_size : index                // 2 * tile_size
    %second_to_last_index = arith.subi %c128, %two_tile_size : index   // all_tiles - 2 * tile_size
    %last_index = arith.subi %c128, %tile_size: index                  // all_tiles - tile_size
    // Here goes all the code that is run on the DM core
    "scf.if"(%is_dm_core) ({
      "snax.mcycle"() : () -> ()
      // Perform the memory transfer on a subview of the memref
      %tiled_A_0 = "memref.subview"(%A, %c0, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L3">, index, index) -> memref<?xi64, strided<[1], offset: ?> , "L3">
      %tiled_B_0 = "memref.subview"(%B, %c0, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L3">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L3">
      %tiled_A_L1_0 = "memref.subview"(%A_L1, %c0, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
      %tiled_B_L1_0 = "memref.subview"(%B_L1, %c0, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
      "memref.copy"(%tiled_A_0, %tiled_A_L1_0) : (memref<?xi64, strided<[1], offset:?>, "L3">, memref<?xi64, strided<[1],offset: ?>, "L1">) -> ()
      "memref.copy"(%tiled_B_0, %tiled_B_L1_0) : (memref<?xi64, strided<[1], offset:?>, "L3">, memref<?xi64, strided<[1],offset: ?>, "L1">) -> ()
      // Perform the second transfer
      "snax.mcycle"() : () -> ()
      "snax.cluster_sync_op"() : () -> ()
      "snax.mcycle"() : () -> ()
      %tiled_A_1 = "memref.subview"(%A, %tile_size, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L3">, index, index) -> memref<?xi64, strided<[1], offset: ?> , "L3">
      %tiled_B_1 = "memref.subview"(%B, %tile_size, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L3">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L3">
      %tiled_A_L1_1 = "memref.subview"(%A_L1, %tile_size, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
      %tiled_B_L1_1 = "memref.subview"(%B_L1, %tile_size, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
      "memref.copy"(%tiled_A_1, %tiled_A_L1_1) : (memref<?xi64, strided<[1], offset:?>, "L3">, memref<?xi64, strided<[1],offset: ?>, "L1">) -> ()
      "memref.copy"(%tiled_B_1, %tiled_B_L1_1) : (memref<?xi64, strided<[1], offset:?>, "L3">, memref<?xi64, strided<[1],offset: ?>, "L1">) -> ()
      "snax.mcycle"() : () -> ()
      "snax.cluster_sync_op"() : () -> ()
      // hot loop, send inputs and outputs
      scf.for %iv = %two_tile_size to %c128 step %tile_size {
          "snax.mcycle"() : () -> ()
          %tiled_A = "memref.subview"(%A, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L3">, index, index) -> memref<?xi64, strided<[1], offset: ?> , "L3">
          %tiled_B = "memref.subview"(%B, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L3">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L3">
          %tiled_A_L1 = "memref.subview"(%A_L1, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
          %tiled_B_L1 = "memref.subview"(%B_L1, %iv, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
          "memref.copy"(%tiled_A, %tiled_A_L1) : (memref<?xi64, strided<[1], offset:?>, "L3">, memref<?xi64, strided<[1],offset: ?>, "L1">) -> ()
          "memref.copy"(%tiled_B, %tiled_B_L1) : (memref<?xi64, strided<[1], offset:?>, "L3">, memref<?xi64, strided<[1],offset: ?>, "L1">) -> ()
          // Transfer 2 tiles ago
          %two_indexes_ago = arith.subi %iv, %two_tile_size : index
          %tiled_D = "memref.subview"(%D, %two_indexes_ago, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L3">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L3">
          %tiled_D_L1 = "memref.subview"(%D_L1, %two_indexes_ago, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
          "memref.copy"(%tiled_D_L1, %tiled_D) : (memref<?xi64, strided<[1], offset: ?>, "L1">, memref<?xi64, strided<[1], offset: ?>, "L3">) -> ()
          // Wait until synchronized with compute core
          "snax.mcycle"() : () -> ()
          "snax.cluster_sync_op"() : () -> ()
          scf.yield
      }
      "snax.mcycle"() : () -> ()
      // Synchronize with compute core
      // Send second to last output
      %tiled_D_0 = "memref.subview"(%D, %second_to_last_index, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L3">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L3">
      %tiled_D_L1_0 = "memref.subview"(%D_L1, %second_to_last_index, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
      "memref.copy"(%tiled_D_L1_0, %tiled_D_0) : (memref<?xi64, strided<[1], offset: ?>, "L1">, memref<?xi64, strided<[1], offset: ?>, "L3">) -> ()
      "snax.mcycle"() : () -> ()
      "snax.cluster_sync_op"() : () -> ()
      "snax.mcycle"() : () -> ()
      // Wait for compute core to finish computing here
      // Send last output
      %tiled_D_1 = "memref.subview"(%D, %last_index, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L3">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L3">
      %tiled_D_L1_1 = "memref.subview"(%D_L1, %last_index, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
      "memref.copy"(%tiled_D_L1_1, %tiled_D_1) : (memref<?xi64, strided<[1], offset: ?>, "L1">, memref<?xi64, strided<[1], offset: ?>, "L3">) -> ()
      "snax.mcycle"() : () -> ()
      "snax.cluster_sync_op"() : () -> ()
      // Send back output from L1 to L3
      scf.yield
    },{
      // Don't do anything if not a dm core
      scf.yield
    }) : (i1) -> ()
    // Here goes all the code that is run on the compute core
    "scf.if"(%is_compute_core) ({
      // Wait for first input tile to come from DM core
      "snax.mcycle"() : () -> ()
      "snax.mcycle"() : () -> ()
      "snax.cluster_sync_op"() : () -> ()
      "snax.mcycle"() : () -> ()
      %tiled_A_L1_0 = "memref.subview"(%A_L1, %c0, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64:-9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
      %tiled_B_L1_0 = "memref.subview"(%B_L1, %c0, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64:-9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
      %tiled_D_L1_0 = "memref.subview"(%D_L1, %c0, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64:-9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
      "linalg.generic"(%tiled_A_L1_0, %tiled_B_L1_0, %tiled_D_L1_0) <{indexing_maps = [affine_map<(n) -> (n)>, affine_map<(n) -> (n)>, affine_map<(n) -> (n)>], iterator_types = [#linalg.iterator_type<parallel>], library_call = "snax_alu", operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg3: i64, %arg4: i64, %arg5: i64):
        %10 = "arith.addi"(%arg3, %arg4) : (i64, i64) -> i64
        "linalg.yield"(%10) : (i64) -> ()
      }) : (memref<?xi64, strided<[1], offset: ?>, "L1">, memref<?xi64, strided<[1], offset: ?>, "L1">, memref<?xi64, strided<[1], offset: ?>, "L1">) -> ()
      "snax.mcycle"() : () -> ()
      "snax.cluster_sync_op"() : () -> ()
      scf.for %iv = %tile_size to %last_index step %tile_size {
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
          "snax.mcycle"() : () -> ()
          "snax.cluster_sync_op"() : () -> ()
      }
      "snax.mcycle"() : () -> ()
      %tiled_A_L1_1 = "memref.subview"(%A_L1, %last_index, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64:-9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
      %tiled_B_L1_1 = "memref.subview"(%B_L1, %last_index, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64:-9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
      %tiled_D_L1_1 = "memref.subview"(%D_L1, %last_index, %tile_size) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64:-9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<128xi64, "L1">, index, index) -> memref<?xi64, strided<[1], offset: ?>, "L1">
      "linalg.generic"(%tiled_A_L1_1, %tiled_B_L1_1, %tiled_D_L1_1) <{indexing_maps = [affine_map<(n) -> (n)>, affine_map<(n) -> (n)>, affine_map<(n) -> (n)>], iterator_types = [#linalg.iterator_type<parallel>], library_call = "snax_alu", operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg3: i64, %arg4: i64, %arg5: i64):
        %10 = "arith.addi"(%arg3, %arg4) : (i64, i64) -> i64
        "linalg.yield"(%10) : (i64) -> ()
      }) : (memref<?xi64, strided<[1], offset: ?>, "L1">, memref<?xi64, strided<[1], offset: ?>, "L1">, memref<?xi64, strided<[1], offset: ?>, "L1">) -> ()
      // Wait for output to go from L1 to L3
      "snax.mcycle"() : () -> ()
      "snax.cluster_sync_op"() : () -> ()
      "snax.mcycle"() : () -> ()
      "snax.mcycle"() : () -> ()
      "snax.cluster_sync_op"() : () -> ()
      "snax.mcycle"() : () -> ()
      "snax.mcycle"() : () -> ()
      scf.yield
    }, {
      scf.yield
      // Don't do anything if not a compute core
    }) : (i1) -> ()
    // return to main routine
    return
}
