// RUN: ./compiler/snax-opt --split-input-file %s -p set-memory-space | filecheck %s

"builtin.module"() ({
  %0 = "memref.get_global"() <{"name" = @constant}> : () -> memref<640xi32>
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = memref.get_global @constant : memref<640xi32, "L3">
// CHECK-NEXT:  }

// -----

"builtin.module"() ({
  %0 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<640xi32>
}) : () -> ()

// CHECK:  builtin.module {
// CHECK-NEXT:    %0 = memref.alloc() {alignment = 64 : i64} : memref<640xi32, "L1">
// CHECK-NEXT:  }

// -----

"builtin.module"() ({
  "func.func"() <{function_type = (memref<64xi32>) -> memref<64xi32>, sym_name = "test", sym_visibility = "public"}> ({
  ^bb0(%arg0: memref<64xi32>):
    "func.return"(%arg0) : (memref<64xi32>) -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:  builtin.module {
// CHECK-NEXT:    func.func public @test(%arg0 : memref<64xi32, "L3">) -> memref<64xi32, "L3"> {
// CHECK-NEXT:      func.return %arg0 : memref<64xi32, "L3">
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

"builtin.module"() ({
  "func.func"() <{function_type = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), sym_name = "simple_mult", sym_visibility = "public"}> ({
  ^bb0(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>):
    "linalg.generic"(%arg0, %arg1, %arg2) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %0 = "arith.muli"(%arg3, %arg4) : (i32, i32) -> i32
      "linalg.yield"(%0) : (i32) -> ()
    }) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:  builtin.module {
// CHECK-NEXT:    func.func public @simple_mult(%arg0 : memref<64xi32, "L3">, %arg1 : memref<64xi32, "L3">, %arg2 : memref<64xi32, "L3">) {
// CHECK-NEXT:      %0 = "memref.memory_space_cast"(%arg0) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
// CHECK-NEXT:      %1 = "memref.memory_space_cast"(%arg1) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
// CHECK-NEXT:      %2 = "memref.memory_space_cast"(%arg2) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
// CHECK-NEXT:      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32, "L1">, memref<64xi32, "L1">) outs(%2 : memref<64xi32, "L1">) {
// CHECK-NEXT:      ^0(%arg3 : i32, %arg4 : i32, %arg5 : i32):
// CHECK-NEXT:        %3 = arith.muli %arg3, %arg4 : i32
// CHECK-NEXT:        linalg.yield %3 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

"builtin.module"() ({
  "func.func"() <{function_type = (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> (), sym_name = "simple_mult", sym_visibility = "public"}> ({
  ^bb0(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>):
    "linalg.generic"(%arg0, %arg1, %arg2) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %0 = "arith.muli"(%arg3, %arg4) : (i32, i32) -> i32
      "linalg.yield"(%0) : (i32) -> ()
    }) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
    "linalg.generic"(%arg0, %arg1, %arg2) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %0 = "arith.muli"(%arg3, %arg4) : (i32, i32) -> i32
      "linalg.yield"(%0) : (i32) -> ()
    }) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()


// CHECK:  builtin.module {
// CHECK-NEXT:    func.func public @simple_mult(%arg0 : memref<64xi32, "L3">, %arg1 : memref<64xi32, "L3">, %arg2 : memref<64xi32, "L3">) {
// CHECK-NEXT:      %0 = "memref.memory_space_cast"(%arg0) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
// CHECK-NEXT:      %1 = "memref.memory_space_cast"(%arg1) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
// CHECK-NEXT:      %2 = "memref.memory_space_cast"(%arg2) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
// CHECK-NEXT:      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32, "L1">, memref<64xi32, "L1">) outs(%2 : memref<64xi32, "L1">) {
// CHECK-NEXT:      ^0(%arg3 : i32, %arg4 : i32, %arg5 : i32):
// CHECK-NEXT:        %3 = arith.muli %arg3, %arg4 : i32
// CHECK-NEXT:        linalg.yield %3 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32, "L1">, memref<64xi32, "L1">) outs(%2 : memref<64xi32, "L1">) {
// CHECK-NEXT:      ^1(%arg3_1 : i32, %arg4_1 : i32, %arg5_1 : i32):
// CHECK-NEXT:        %4 = arith.muli %arg3_1, %arg4_1 : i32
// CHECK-NEXT:        linalg.yield %4 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

"builtin.module"() ({
  "func.func"() <{function_type = (memref<?xi32>, memref<?xi32>, memref<?xi32>) -> (), sym_name = "simple_mult", sym_visibility = "public"}> ({
  ^bb0(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>):
    "linalg.generic"(%arg0, %arg1, %arg2) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %0 = "arith.muli"(%arg3, %arg4) : (i32, i32) -> i32
      "linalg.yield"(%0) : (i32) -> ()
    }) : (memref<?xi32>, memref<?xi32>, memref<?xi32>) -> ()
    "linalg.generic"(%arg0, %arg1, %arg2) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %0 = "arith.muli"(%arg3, %arg4) : (i32, i32) -> i32
      "linalg.yield"(%0) : (i32) -> ()
    }) : (memref<?xi32>, memref<?xi32>, memref<?xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:  builtin.module {
// CHECK-NEXT:    func.func public @simple_mult(%arg0 : memref<?xi32, "L3">, %arg1 : memref<?xi32, "L3">, %arg2 : memref<?xi32, "L3">) {
// CHECK-NEXT:      %0 = "memref.memory_space_cast"(%arg0) : (memref<?xi32, "L3">) -> memref<?xi32, "L1">
// CHECK-NEXT:      %1 = "memref.memory_space_cast"(%arg1) : (memref<?xi32, "L3">) -> memref<?xi32, "L1">
// CHECK-NEXT:      %2 = "memref.memory_space_cast"(%arg2) : (memref<?xi32, "L3">) -> memref<?xi32, "L1">
// CHECK-NEXT:      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<?xi32, "L1">, memref<?xi32, "L1">) outs(%2 : memref<?xi32, "L1">) {
// CHECK-NEXT:      ^0(%arg3 : i32, %arg4 : i32, %arg5 : i32):
// CHECK-NEXT:        %3 = arith.muli %arg3, %arg4 : i32
// CHECK-NEXT:        linalg.yield %3 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<?xi32, "L1">, memref<?xi32, "L1">) outs(%2 : memref<?xi32, "L1">) {
// CHECK-NEXT:      ^1(%arg3_1 : i32, %arg4_1 : i32, %arg5_1 : i32):
// CHECK-NEXT:        %4 = arith.muli %arg3_1, %arg4_1 : i32
// CHECK-NEXT:        linalg.yield %4 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

func.func @gemm(%arg0 : memref<16x16xi8>, %arg1 : memref<16x16xi8>, %arg2 : memref<16x16xi32>) -> memref<16x16xi32> {
  %0 = arith.constant 0 : i32
  %1 = memref.get_global @_static_const_0 : memref<16x16xi32>
  "stream.streaming_region"(%arg0, %arg1, %arg2, %1) <{"operandSegmentSizes" = array<i32: 3, 1>, "patterns" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
  ^0(%arg3 : !stream.stream<i8>, %arg4 : !stream.stream<i8>, %arg5 : !stream.stream<i32>, %arg6 : !stream.stream<i32>):
    %4 = "test.op"(%arg3, %arg4, %arg5) : (!stream.stream<i8>, !stream.stream<i8>, !stream.stream<i32>) -> !stream.stream<i32>
    stream.yield %4 : !stream.stream<i32>
  }) : (memref<16x16xi8>, memref<16x16xi8>, memref<16x16xi32>, memref<16x16xi32>) -> ()
  func.return %1 : memref<16x16xi32>
}

// CHECK:  builtin.module {
// CHECK-NEXT:    func.func @gemm(%arg0 : memref<16x16xi8, "L3">, %arg1 : memref<16x16xi8, "L3">, %arg2 : memref<16x16xi32, "L3">) -> memref<16x16xi32, "L3"> {
// CHECK-NEXT:      %0 = arith.constant 0 : i32
// CHECK-NEXT:      %1 = memref.get_global @_static_const_0 : memref<16x16xi32, "L3">
// CHECK-NEXT:      %2 = "memref.memory_space_cast"(%arg0) : (memref<16x16xi8, "L3">) -> memref<16x16xi8, "L1">
// CHECK-NEXT:      %3 = "memref.memory_space_cast"(%arg1) : (memref<16x16xi8, "L3">) -> memref<16x16xi8, "L1">
// CHECK-NEXT:      %4 = "memref.memory_space_cast"(%arg2) : (memref<16x16xi32, "L3">) -> memref<16x16xi32, "L1">
// CHECK-NEXT:      %5 = "memref.memory_space_cast"(%1) : (memref<16x16xi32, "L3">) -> memref<16x16xi32, "L1">
// CHECK-NEXT:      "stream.streaming_region"(%2, %3, %4, %5) <{operandSegmentSizes = array<i32: 3, 1>, patterns = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
// CHECK-NEXT:      ^0(%arg3 : !stream.stream<i8>, %arg4 : !stream.stream<i8>, %arg5 : !stream.stream<i32>, %arg6 : !stream.stream<i32>):
// CHECK-NEXT:        %6 = "test.op"(%arg3, %arg4, %arg5) : (!stream.stream<i8>, !stream.stream<i8>, !stream.stream<i32>) -> !stream.stream<i32>
// CHECK-NEXT:        stream.yield %6 : !stream.stream<i32>
// CHECK-NEXT:      }) : (memref<16x16xi8, "L1">, memref<16x16xi8, "L1">, memref<16x16xi32, "L1">, memref<16x16xi32, "L1">) -> ()
// CHECK-NEXT:      func.return %1 : memref<16x16xi32, "L3">
// CHECK-NEXT:    }
// CHECK-NEXT:  }
