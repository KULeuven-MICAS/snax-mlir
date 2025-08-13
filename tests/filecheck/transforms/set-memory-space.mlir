// RUN: snax-opt --split-input-file %s -p set-memory-space | filecheck %s

%0 = memref.get_global @constant : memref<640xi32>

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = memref.get_global @constant : memref<640xi32, "L3">
// CHECK-NEXT:  }

// -----

%0 = memref.alloc() {alignment = 64 : i64} : memref<640xi32>

// CHECK:  builtin.module {
// CHECK-NEXT:    %0 = memref.alloc() {alignment = 64 : i64} : memref<640xi32, "L1">
// CHECK-NEXT:  }

// -----

func.func public @test(%arg0 : memref<64xi32>) -> memref<64xi32> {
  func.return %arg0 : memref<64xi32>
}

// CHECK:  builtin.module {
// CHECK-NEXT:    func.func public @test(%arg0 : memref<64xi32, "L3">) -> memref<64xi32, "L3"> {
// CHECK-NEXT:      func.return %arg0 : memref<64xi32, "L3">
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

func.func public @test() -> (memref<64xi32>, memref<64xi32>) {
  %0 = memref.get_global @constant : memref<64xi32>
  %1 = memref.alloc() {alignment = 64 : i64} : memref<64xi32>
  func.return %0, %1: memref<64xi32>, memref<64xi32>
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func public @test() -> (memref<64xi32, "L3">, memref<64xi32, "L3">) {
// CHECK-NEXT:     %0 = memref.get_global @constant : memref<64xi32, "L3">
// CHECK-NEXT:     %1 = memref.alloc() {alignment = 64 : i64} : memref<64xi32, "L1">
// CHECK-NEXT:     %2 = "memref.memory_space_cast"(%1) : (memref<64xi32, "L1">) -> memref<64xi32, "L3">
// CHECK-NEXT:     func.return %0, %2 : memref<64xi32, "L3">, memref<64xi32, "L3">
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

func.func public @simple_mult(%arg0 : memref<64xi32>, %arg1 : memref<64xi32>, %arg2 : memref<64xi32>) {
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xi32>, memref<64xi32>) outs(%arg2 : memref<64xi32>) {
  ^bb0(%arg3 : i32, %arg4 : i32, %arg5 : i32):
    %0 = arith.muli %arg3, %arg4 : i32
    linalg.yield %0 : i32
  }
  func.return
}

// CHECK:  builtin.module {
// CHECK-NEXT:    func.func public @simple_mult(%arg0 : memref<64xi32, "L3">, %arg1 : memref<64xi32, "L3">, %arg2 : memref<64xi32, "L3">) {
// CHECK-NEXT:      %0 = "memref.memory_space_cast"(%arg0) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
// CHECK-NEXT:      %1 = "memref.memory_space_cast"(%arg1) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
// CHECK-NEXT:      %2 = "memref.memory_space_cast"(%arg2) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
// CHECK-NEXT:      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32, "L1">, memref<64xi32, "L1">) outs(%2 : memref<64xi32, "L1">) {
// CHECK-NEXT:      ^bb0(%arg3 : i32, %arg4 : i32, %arg5 : i32):
// CHECK-NEXT:        %3 = arith.muli %arg3, %arg4 : i32
// CHECK-NEXT:        linalg.yield %3 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

func.func public @simple_mult(%arg0 : memref<64xi32>, %arg1 : memref<64xi32>, %arg2 : memref<64xi32>) {
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xi32>, memref<64xi32>) outs(%arg2 : memref<64xi32>) {
  ^bb0(%arg3 : i32, %arg4 : i32, %arg5 : i32):
    %0 = arith.muli %arg3, %arg4 : i32
    linalg.yield %0 : i32
  }
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xi32>, memref<64xi32>) outs(%arg2 : memref<64xi32>) {
  ^bb1(%arg3_1 : i32, %arg4_1 : i32, %arg5_1 : i32):
    %1 = arith.muli %arg3_1, %arg4_1 : i32
    linalg.yield %1 : i32
  }
  func.return
}

// CHECK:  builtin.module {
// CHECK-NEXT:    func.func public @simple_mult(%arg0 : memref<64xi32, "L3">, %arg1 : memref<64xi32, "L3">, %arg2 : memref<64xi32, "L3">) {
// CHECK-NEXT:      %0 = "memref.memory_space_cast"(%arg0) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
// CHECK-NEXT:      %1 = "memref.memory_space_cast"(%arg1) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
// CHECK-NEXT:      %2 = "memref.memory_space_cast"(%arg2) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
// CHECK-NEXT:      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32, "L1">, memref<64xi32, "L1">) outs(%2 : memref<64xi32, "L1">) {
// CHECK-NEXT:      ^bb0(%arg3 : i32, %arg4 : i32, %arg5 : i32):
// CHECK-NEXT:        %3 = arith.muli %arg3, %arg4 : i32
// CHECK-NEXT:        linalg.yield %3 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32, "L1">, memref<64xi32, "L1">) outs(%2 : memref<64xi32, "L1">) {
// CHECK-NEXT:      ^bb1(%arg3_1 : i32, %arg4_1 : i32, %arg5_1 : i32):
// CHECK-NEXT:        %4 = arith.muli %arg3_1, %arg4_1 : i32
// CHECK-NEXT:        linalg.yield %4 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

func.func public @simple_mult(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xi32>) {
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xi32>, memref<?xi32>) outs(%arg2 : memref<?xi32>) {
  ^bb0(%arg3 : i32, %arg4 : i32, %arg5 : i32):
    %0 = arith.muli %arg3, %arg4 : i32
    linalg.yield %0 : i32
  }
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xi32>, memref<?xi32>) outs(%arg2 : memref<?xi32>) {
  ^bb1(%arg3_1 : i32, %arg4_1 : i32, %arg5_1 : i32):
    %1 = arith.muli %arg3_1, %arg4_1 : i32
    linalg.yield %1 : i32
  }
  func.return
}

// CHECK:  builtin.module {
// CHECK-NEXT:    func.func public @simple_mult(%arg0 : memref<?xi32, "L3">, %arg1 : memref<?xi32, "L3">, %arg2 : memref<?xi32, "L3">) {
// CHECK-NEXT:      %0 = "memref.memory_space_cast"(%arg0) : (memref<?xi32, "L3">) -> memref<?xi32, "L1">
// CHECK-NEXT:      %1 = "memref.memory_space_cast"(%arg1) : (memref<?xi32, "L3">) -> memref<?xi32, "L1">
// CHECK-NEXT:      %2 = "memref.memory_space_cast"(%arg2) : (memref<?xi32, "L3">) -> memref<?xi32, "L1">
// CHECK-NEXT:      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<?xi32, "L1">, memref<?xi32, "L1">) outs(%2 : memref<?xi32, "L1">) {
// CHECK-NEXT:      ^bb0(%arg3 : i32, %arg4 : i32, %arg5 : i32):
// CHECK-NEXT:        %3 = arith.muli %arg3, %arg4 : i32
// CHECK-NEXT:        linalg.yield %3 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<?xi32, "L1">, memref<?xi32, "L1">) outs(%2 : memref<?xi32, "L1">) {
// CHECK-NEXT:      ^bb1(%arg3_1 : i32, %arg4_1 : i32, %arg5_1 : i32):
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
  "dart.operation"(%arg0, %arg1, %arg2, %1) <{"operandSegmentSizes" = array<i32: 3, 1>, "patterns" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
  ^bb0(%arg3 : !dart.stream<i8>, %arg4 : !dart.stream<i8>, %arg5 : !dart.stream<i32>, %arg6 : !dart.stream<i32>):
    %4 = "test.op"(%arg3, %arg4, %arg5) : (!dart.stream<i8>, !dart.stream<i8>, !dart.stream<i32>) -> !dart.stream<i32>
    dart.yield %4 : !dart.stream<i32>
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
// CHECK-NEXT:      "dart.operation"(%2, %3, %4, %5) <{operandSegmentSizes = array<i32: 3, 1>, patterns = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
// CHECK-NEXT:      ^bb0(%arg3 : !dart.stream<i8>, %arg4 : !dart.stream<i8>, %arg5 : !dart.stream<i32>, %arg6 : !dart.stream<i32>):
// CHECK-NEXT:        %6 = "test.op"(%arg3, %arg4, %arg5) : (!dart.stream<i8>, !dart.stream<i8>, !dart.stream<i32>) -> !dart.stream<i32>
// CHECK-NEXT:        dart.yield %6 : !dart.stream<i32>
// CHECK-NEXT:      }) : (memref<16x16xi8, "L1">, memref<16x16xi8, "L1">, memref<16x16xi32, "L1">, memref<16x16xi32, "L1">) -> ()
// CHECK-NEXT:      func.return %1 : memref<16x16xi32, "L3">
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

%0 = memref.alloc() {alignment = 64 : i64} : memref<640xi32>
%1 = memref.subview %0[0] [16] [2] : memref<640xi32> to memref<16xi32, strided<[2], offset: 0>>

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = memref.alloc() {alignment = 64 : i64} : memref<640xi32, "L1">
// CHECK-NEXT:   %1 = memref.subview %0[0] [16] [2] : memref<640xi32, "L1"> to memref<16xi32, strided<[2]>, "L1">
// CHECK-NEXT: }
