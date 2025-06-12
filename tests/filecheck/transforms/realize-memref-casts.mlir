// RUN: snax-opt --split-input-file %s -p realize-memref-casts | filecheck %s

%0 = "test.op"() : () -> memref<64xi32, "L3">
%1 = "memref.memory_space_cast"(%0) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
"test.op"(%1) : (memref<64xi32, "L1">) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> memref<64xi32, "L3">
// CHECK-NEXT:   %1 = memref.alloc() {alignment = 64 : i64} : memref<64xi32, "L1">
// CHECK-NEXT:   "memref.copy"(%0, %1) : (memref<64xi32, "L3">, memref<64xi32, "L1">) -> ()
// CHECK-NEXT:   "test.op"(%1) : (memref<64xi32, "L1">) -> ()
// CHECK-NEXT:   "memref.copy"(%1, %0) : (memref<64xi32, "L1">, memref<64xi32, "L3">) -> ()
// CHECK-NEXT: }

// -----

%0 = "test.op"() : () -> memref<64xi32, "L1">
%1 = "snax.layout_cast"(%0) : (memref<64xi32, "L1">) -> memref<64xi32, strided<[1, 8]>, "L1">
"test.op"(%1) : (memref<64xi32, strided<[1, 8]>, "L1">) -> ()



// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> memref<64xi32, "L1">
// CHECK-NEXT:   %1 = memref.alloc() {alignment = 64 : i64} : memref<64xi32, strided<[1, 8]>, "L1">
// CHECK-NEXT:   "memref.copy"(%0, %1) : (memref<64xi32, "L1">, memref<64xi32, strided<[1, 8]>, "L1">) -> ()
// CHECK-NEXT:   "test.op"(%1) : (memref<64xi32, strided<[1, 8]>, "L1">) -> ()
// CHECK-NEXT:   "memref.copy"(%1, %0) : (memref<64xi32, strided<[1, 8]>, "L1">, memref<64xi32, "L1">) -> ()
// CHECK-NEXT: }

// -----

%0 = "test.op"() : () -> memref<64xi32, "L3">
%1 = "memref.memory_space_cast"(%0) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
%2 = "snax.layout_cast"(%1) : (memref<64xi32, "L1">) -> memref<64xi32, strided<[1, 8]>, "L1">


// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> memref<64xi32, "L3">
// CHECK-NEXT: }

// -----

func.func public @simple_mult(%arg0 : memref<64xi32, "L3">, %arg1 : memref<64xi32, "L3">, %arg2 : memref<64xi32, "L3">) {
  %0 = "memref.memory_space_cast"(%arg0) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
  %1 = "memref.memory_space_cast"(%arg1) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
  %2 = "memref.memory_space_cast"(%arg2) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32, "L1">, memref<64xi32, "L1">) outs(%2 : memref<64xi32, "L1">) {
  ^0(%arg3 : i32, %arg4 : i32, %arg5 : i32):
    %3 = arith.muli %arg3, %arg4 : i32
    linalg.yield %3 : i32
  }
  func.return
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%arg0 : memref<64xi32, "L3">, %arg1 : memref<64xi32, "L3">, %arg2 : memref<64xi32, "L3">) {
// CHECK-NEXT:     %0 = memref.alloc() {alignment = 64 : i64} : memref<64xi32, "L1">
// CHECK-NEXT:     %1 = memref.alloc() {alignment = 64 : i64} : memref<64xi32, "L1">
// CHECK-NEXT:     %2 = memref.alloc() {alignment = 64 : i64} : memref<64xi32, "L1">
// CHECK-NEXT:     "memref.copy"(%arg1, %1) : (memref<64xi32, "L3">, memref<64xi32, "L1">) -> ()
// CHECK-NEXT:     "memref.copy"(%arg0, %0) : (memref<64xi32, "L3">, memref<64xi32, "L1">) -> ()
// CHECK-NEXT:     linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32, "L1">, memref<64xi32, "L1">) outs(%2 : memref<64xi32, "L1">) {
// CHECK-NEXT:     ^0(%arg3 : i32, %arg4 : i32, %arg5 : i32):
// CHECK-NEXT:       %3 = arith.muli %arg3, %arg4 : i32
// CHECK-NEXT:       linalg.yield %3 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     "memref.copy"(%2, %arg2) : (memref<64xi32, "L1">, memref<64xi32, "L3">) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

func.func public @simple_mult(%arg0 : memref<64xi32, "L3">, %arg1 : memref<64xi32, "L3">, %arg2 : memref<64xi32, "L3">) {
  %0 = "memref.memory_space_cast"(%arg0) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
  %1 = "memref.memory_space_cast"(%arg1) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
  %2 = "memref.memory_space_cast"(%arg2) : (memref<64xi32, "L3">) -> memref<64xi32, "L1">
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32, "L1">, memref<64xi32, "L1">) outs(%2 : memref<64xi32, "L1">) {
  ^0(%arg3 : i32, %arg4 : i32, %arg5 : i32):
    %3 = arith.muli %arg3, %arg4 : i32
    linalg.yield %3 : i32
  }
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32, "L1">, memref<64xi32, "L1">) outs(%2 : memref<64xi32, "L1">) {
  ^1(%arg3_1 : i32, %arg4_1 : i32, %arg5_1 : i32):
    %4 = arith.muli %arg3_1, %arg4_1 : i32
    linalg.yield %4 : i32
  }
  func.return
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%arg0 : memref<64xi32, "L3">, %arg1 : memref<64xi32, "L3">, %arg2 : memref<64xi32, "L3">) {
// CHECK-NEXT:     %0 = memref.alloc() {alignment = 64 : i64} : memref<64xi32, "L1">
// CHECK-NEXT:     %1 = memref.alloc() {alignment = 64 : i64} : memref<64xi32, "L1">
// CHECK-NEXT:     %2 = memref.alloc() {alignment = 64 : i64} : memref<64xi32, "L1">
// CHECK-NEXT:     "memref.copy"(%arg1, %1) : (memref<64xi32, "L3">, memref<64xi32, "L1">) -> ()
// CHECK-NEXT:     "memref.copy"(%arg0, %0) : (memref<64xi32, "L3">, memref<64xi32, "L1">) -> ()
// CHECK-NEXT:     linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32, "L1">, memref<64xi32, "L1">) outs(%2 : memref<64xi32, "L1">) {
// CHECK-NEXT:     ^0(%arg3 : i32, %arg4 : i32, %arg5 : i32):
// CHECK-NEXT:       %3 = arith.muli %arg3, %arg4 : i32
// CHECK-NEXT:       linalg.yield %3 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<64xi32, "L1">, memref<64xi32, "L1">) outs(%2 : memref<64xi32, "L1">) {
// CHECK-NEXT:     ^1(%arg3_1 : i32, %arg4_1 : i32, %arg5_1 : i32):
// CHECK-NEXT:       %4 = arith.muli %arg3_1, %arg4_1 : i32
// CHECK-NEXT:       linalg.yield %4 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     "memref.copy"(%2, %arg2) : (memref<64xi32, "L1">, memref<64xi32, "L3">) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

func.func public @simple_mult(%arg0 : memref<?xi32, "L3">, %arg1 : memref<?xi32, "L3">, %arg2 : memref<?xi32, "L3">) {
  %0 = "memref.memory_space_cast"(%arg0) : (memref<?xi32, "L3">) -> memref<?xi32, "L1">
  %1 = "memref.memory_space_cast"(%arg1) : (memref<?xi32, "L3">) -> memref<?xi32, "L1">
  %2 = "memref.memory_space_cast"(%arg2) : (memref<?xi32, "L3">) -> memref<?xi32, "L1">
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<?xi32, "L1">, memref<?xi32, "L1">) outs(%2 : memref<?xi32, "L1">) {
  ^0(%arg3 : i32, %arg4 : i32, %arg5 : i32):
    %3 = arith.muli %arg3, %arg4 : i32
    linalg.yield %3 : i32
  }
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<?xi32, "L1">, memref<?xi32, "L1">) outs(%2 : memref<?xi32, "L1">) {
  ^1(%arg3_1 : i32, %arg4_1 : i32, %arg5_1 : i32):
    %4 = arith.muli %arg3_1, %arg4_1 : i32
    linalg.yield %4 : i32
  }
  func.return
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%arg0 : memref<?xi32, "L3">, %arg1 : memref<?xi32, "L3">, %arg2 : memref<?xi32, "L3">) {
// CHECK-NEXT:     %0 = arith.constant 0 : index
// CHECK-NEXT:     %1 = "memref.dim"(%arg0, %0) : (memref<?xi32, "L3">, index) -> index
// CHECK-NEXT:     %2 = memref.alloc(%1) {alignment = 64 : i64} : memref<?xi32, "L1">
// CHECK-NEXT:     %3 = arith.constant 0 : index
// CHECK-NEXT:     %4 = "memref.dim"(%arg1, %3) : (memref<?xi32, "L3">, index) -> index
// CHECK-NEXT:     %5 = memref.alloc(%4) {alignment = 64 : i64} : memref<?xi32, "L1">
// CHECK-NEXT:     %6 = arith.constant 0 : index
// CHECK-NEXT:     %7 = "memref.dim"(%arg2, %6) : (memref<?xi32, "L3">, index) -> index
// CHECK-NEXT:     %8 = memref.alloc(%7) {alignment = 64 : i64} : memref<?xi32, "L1">
// CHECK-NEXT:     "memref.copy"(%arg1, %5) : (memref<?xi32, "L3">, memref<?xi32, "L1">) -> ()
// CHECK-NEXT:     "memref.copy"(%arg0, %2) : (memref<?xi32, "L3">, memref<?xi32, "L1">) -> ()
// CHECK-NEXT:     linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2, %5 : memref<?xi32, "L1">, memref<?xi32, "L1">) outs(%8 : memref<?xi32, "L1">) {
// CHECK-NEXT:     ^0(%arg3 : i32, %arg4 : i32, %arg5 : i32):
// CHECK-NEXT:       %9 = arith.muli %arg3, %arg4 : i32
// CHECK-NEXT:       linalg.yield %9 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2, %5 : memref<?xi32, "L1">, memref<?xi32, "L1">) outs(%8 : memref<?xi32, "L1">) {
// CHECK-NEXT:     ^1(%arg3_1 : i32, %arg4_1 : i32, %arg5_1 : i32):
// CHECK-NEXT:       %10 = arith.muli %arg3_1, %arg4_1 : i32
// CHECK-NEXT:       linalg.yield %10 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     "memref.copy"(%8, %arg2) : (memref<?xi32, "L1">, memref<?xi32, "L3">) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

func.func @mnist(%arg0 : memref<?x128xi8, "L3">, %arg1 : memref<128x128xi8, "L3">, %arg2 : memref<?x128xi32, "L3">) -> memref<?x128xi32, "L3"> {
  %0 = arith.constant 0 : i32
  %1 = arith.constant 0 : i32
  %2 = "memref.memory_space_cast"(%arg0) : (memref<?x128xi8, "L3">) -> memref<?x128xi8, "L1">
  %3 = "memref.memory_space_cast"(%arg1) : (memref<128x128xi8, "L3">) -> memref<128x128xi8, "L1">
  %4 = "memref.memory_space_cast"(%arg2) : (memref<?x128xi32, "L3">) -> memref<?x128xi32, "L1">
  %5 = "snax.layout_cast"(%2) : (memref<?x128xi8, "L1">) -> memref<?x128xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
  %6 = "snax.layout_cast"(%3) : (memref<128x128xi8, "L1">) -> memref<128x128xi8, #tsl.tsl<[?, 8] -> (256, 1), [?, 8] -> (?, 8), offset: 64>, "L1">
  %7 = "snax.layout_cast"(%4) : (memref<?x128xi32, "L1">) -> memref<?x128xi32, #tsl.tsl<[?, 8] -> (256, 4), [?, 8] -> (?, 32)>, "L1">
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], library_call = "snax_gemm"} ins(%5, %6, %0, %1 : memref<?x128xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">, memref<128x128xi8, #tsl.tsl<[?, 8] -> (256, 1), [?, 8] -> (?, 8), offset: 64>, "L1">, i32, i32) outs(%7 : memref<?x128xi32, #tsl.tsl<[?, 8] -> (256, 4), [?, 8] -> (?, 32)>, "L1">) {
  ^0(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
    %8 = arith.extsi %arg3 : i8 to i32
    %9 = arith.subi %8, %arg5 : i32
    %10 = arith.extsi %arg4 : i8 to i32
    %11 = arith.subi %10, %arg6 : i32
    %12 = arith.muli %9, %11 : i32
    %13 = arith.addi %arg7, %12 : i32
    linalg.yield %13 : i32
  }
  func.return %arg2 : memref<?x128xi32, "L3">
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @mnist(%arg0 : memref<?x128xi8, "L3">, %arg1 : memref<128x128xi8, "L3">, %arg2 : memref<?x128xi32, "L3">) -> memref<?x128xi32, "L3"> {
// CHECK-NEXT:     %0 = arith.constant 0 : i32
// CHECK-NEXT:     %1 = arith.constant 0 : i32
// CHECK-NEXT:     %2 = "memref.memory_space_cast"(%arg0) : (memref<?x128xi8, "L3">) -> memref<?x128xi8, "L1">
// CHECK-NEXT:     %3 = "memref.memory_space_cast"(%arg1) : (memref<128x128xi8, "L3">) -> memref<128x128xi8, "L1">
// CHECK-NEXT:     %4 = "memref.memory_space_cast"(%arg2) : (memref<?x128xi32, "L3">) -> memref<?x128xi32, "L1">
// CHECK-NEXT:     %5 = arith.constant 0 : index
// CHECK-NEXT:     %6 = "memref.dim"(%arg0, %5) : (memref<?x128xi8, "L3">, index) -> index
// CHECK-NEXT:     %7 = memref.alloc(%6) {alignment = 64 : i64} : memref<?x128xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
// CHECK-NEXT:     %8 = memref.alloc() {alignment = 64 : i64} : memref<128x128xi8, #tsl.tsl<[?, 8] -> (256, 1), [?, 8] -> (?, 8), offset: 64>, "L1">
// CHECK-NEXT:     %9 = arith.constant 0 : index
// CHECK-NEXT:     %10 = "memref.dim"(%arg2, %9) : (memref<?x128xi32, "L3">, index) -> index
// CHECK-NEXT:     %11 = memref.alloc(%10) {alignment = 64 : i64} : memref<?x128xi32, #tsl.tsl<[?, 8] -> (256, 4), [?, 8] -> (?, 32)>, "L1">
// CHECK-NEXT:     "memref.copy"(%arg1, %8) : (memref<128x128xi8, "L3">, memref<128x128xi8, #tsl.tsl<[?, 8] -> (256, 1), [?, 8] -> (?, 8), offset: 64>, "L1">) -> ()
// CHECK-NEXT:     "memref.copy"(%arg0, %7) : (memref<?x128xi8, "L3">, memref<?x128xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
// CHECK-NEXT:     linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], library_call = "snax_gemm"} ins(%7, %8, %0, %1 : memref<?x128xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">, memref<128x128xi8, #tsl.tsl<[?, 8] -> (256, 1), [?, 8] -> (?, 8), offset: 64>, "L1">, i32, i32) outs(%11 : memref<?x128xi32, #tsl.tsl<[?, 8] -> (256, 4), [?, 8] -> (?, 32)>, "L1">) {
// CHECK-NEXT:     ^0(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
// CHECK-NEXT:       %12 = arith.extsi %arg3 : i8 to i32
// CHECK-NEXT:       %13 = arith.subi %12, %arg5 : i32
// CHECK-NEXT:       %14 = arith.extsi %arg4 : i8 to i32
// CHECK-NEXT:       %15 = arith.subi %14, %arg6 : i32
// CHECK-NEXT:       %16 = arith.muli %13, %15 : i32
// CHECK-NEXT:       %17 = arith.addi %arg7, %16 : i32
// CHECK-NEXT:       linalg.yield %17 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     "memref.copy"(%11, %arg2) : (memref<?x128xi32, #tsl.tsl<[?, 8] -> (256, 4), [?, 8] -> (?, 32)>, "L1">, memref<?x128xi32, "L3">) -> ()
// CHECK-NEXT:     func.return %arg2 : memref<?x128xi32, "L3">
// CHECK-NEXT:   }
// CHECK-NEXT: }
// -----

%0 = "test.op"() : () -> memref<64xi32, "L1">
%1 = "memref.memory_space_cast"(%0) : (memref<64xi32, "L1">) -> memref<64xi32, "L1">
"test.op"(%1) : (memref<64xi32, "L1">) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> memref<64xi32, "L1">
// CHECK-NEXT:   "test.op"(%0) : (memref<64xi32, "L1">) -> ()
// CHECK-NEXT: }

// -----

%0 = "test.op"() : () -> memref<64xi32, "L1">
%1 = "snax.layout_cast"(%0) : (memref<64xi32, "L1">) -> memref<64xi32, "L1">
"test.op"(%1) : (memref<64xi32, "L1">) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> memref<64xi32, "L1">
// CHECK-NEXT:   "test.op"(%0) : (memref<64xi32, "L1">) -> ()
// CHECK-NEXT: }

// -----

%0 = "test.op"() : () -> memref<64xi32, "L1">
%1 = "memref.memory_space_cast"(%0) : (memref<64xi32, "L1">) -> memref<64xi32, "L1">
%2 = "snax.layout_cast"(%1) : (memref<64xi32, "L1">) -> memref<64xi32, "L1">
"test.op"(%2) : (memref<64xi32, "L1">) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> memref<64xi32, "L1">
// CHECK-NEXT:   "test.op"(%0) : (memref<64xi32, "L1">) -> ()
// CHECK-NEXT: }

// -----

%0 = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : memref<4x4xi32, "L1">
%1 = "snax.layout_cast"(%0) : (memref<4x4xi32, "L1">) -> memref<4x4xi32, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">
"test.op"(%1) : (memref<4x4xi32, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = arith.constant dense<[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]> : memref<4x4xi32, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">
// CHECK-NEXT:   "test.op"(%0) : (memref<4x4xi32, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">) -> ()
// CHECK-NEXT: }

// -----

%0 = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : memref<4x4xi8, "L1">
%1 = "snax.layout_cast"(%0) : (memref<4x4xi8, "L1">) -> memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">
"test.op"(%1) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = arith.constant dense<[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]> : memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">
// CHECK-NEXT:   "test.op"(%0) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">) -> ()
// CHECK-NEXT: }


// -----

"memref.global"() <{alignment = 64 : i64, constant, initial_value = dense<[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]> : tensor<4x4xi8>, sym_name = "global", sym_visibility = "private", type = memref<4x4xi8>}> : () -> ()
%0 = memref.get_global @global : memref<4x4xi8, "L3">
%1 = "snax.layout_cast"(%0) : (memref<4x4xi8, "L3">) -> memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L3">
"test.op"(%1) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L3">) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = memref.get_global @global_transformed : memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L3">
// CHECK-NEXT:   "test.op"(%0) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L3">) -> ()
// CHECK-NEXT:   "memref.global"() <{sym_name = "global_transformed", type = memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>>, initial_value = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi8>, sym_visibility = "private", constant, alignment = 64 : i64}> : () -> ()
// CHECK-NEXT: }

// -----

"memref.global"() <{alignment = 64 : i64, constant, initial_value, sym_name = "global", sym_visibility = "private", type = memref<4x4xi8>}> : () -> ()
%0 = memref.get_global @global : memref<4x4xi8, "L3">
%1 = "snax.layout_cast"(%0) : (memref<4x4xi8, "L3">) -> memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L3">
"test.op"(%1) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L3">) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = memref.get_global @global_transformed : memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L3">
// CHECK-NEXT:   "test.op"(%0) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L3">) -> ()
// CHECK-NEXT:   "memref.global"() <{sym_name = "global_transformed", type = memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>>, initial_value, sym_visibility = "private", constant, alignment = 64 : i64}> : () -> ()
// CHECK-NEXT: }


// -----

// an alloc with a layout cast

%0 = memref.alloc() {alignment = 64 : i64} : memref<4x4xi8, "L1">
%1 = "snax.layout_cast"(%0) : (memref<4x4xi8, "L1">) -> memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">
"test.op"(%1) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">) -> ()

// the alloc is replaced by a new alloc with the transformed layout

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = memref.alloc() {alignment = 64 : i64} : memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">
// CHECK-NEXT:   "test.op"(%0) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">) -> ()
// CHECK-NEXT: }

// -----

// an alloc with a layout cast, but the alloc is used in a different op

%0 = memref.alloc() {alignment = 64 : i64} : memref<4x4xi8, "L1">
"test.op"(%0) : (memref<4x4xi8, "L1">) -> ()
%1 = "snax.layout_cast"(%0) : (memref<4x4xi8, "L1">) -> memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">
"test.op"(%1) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">) -> ()

// the alloc is unchanged as the test op needs the original layout

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = memref.alloc() {alignment = 64 : i64} : memref<4x4xi8, "L1">
// CHECK-NEXT:   "test.op"(%0) : (memref<4x4xi8, "L1">) -> ()
// CHECK-NEXT:   %1 = memref.alloc() {alignment = 64 : i64} : memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">
// CHECK-NEXT:   "memref.copy"(%0, %1) : (memref<4x4xi8, "L1">, memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">) -> ()
// CHECK-NEXT:   "test.op"(%1) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">) -> ()
// CHECK-NEXT:   "memref.copy"(%1, %0) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">, memref<4x4xi8, "L1">) -> ()
// CHECK-NEXT: }

// -----


// an alloc with two equal layout casts

%0 = memref.alloc() {alignment = 64 : i64} : memref<4x4xi8, "L1">
%1 = "snax.layout_cast"(%0) : (memref<4x4xi8, "L1">) -> memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">
"test.op"(%1) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">) -> ()
%2 = "snax.layout_cast"(%0) : (memref<4x4xi8, "L1">) -> memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">
"test.op"(%2) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">) -> ()

// the alloc is replaced by a new alloc with the transformed layout

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = memref.alloc() {alignment = 64 : i64} : memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">
// CHECK-NEXT:   "test.op"(%0) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">) -> ()
// CHECK-NEXT:   "test.op"(%0) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">) -> ()
// CHECK-NEXT: }

// -----

// an alloc with two different layout casts

%0 = memref.alloc() {alignment = 64 : i64} : memref<4x4xi8, "L1">
%1 = "snax.layout_cast"(%0) : (memref<4x4xi8, "L1">) -> memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">
"test.op"(%1) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">) -> ()
%2 = "snax.layout_cast"(%0) : (memref<4x4xi8, "L1">) -> memref<4x4xi8, #tsl.tsl<[2, 2] -> (10, 2), [2, 2] -> (4, 1)>, "L1">
"test.op"(%2) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (10, 2), [2, 2] -> (4, 1)>, "L1">) -> ()

// the alloc is replaced by a new alloc with the transformed layout, a transformation is applied for the first cast, but not the second one

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = memref.alloc() {alignment = 64 : i64} : memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">
// CHECK-NEXT:   "test.op"(%0) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">) -> ()
// CHECK-NEXT:   %1 = memref.alloc() {alignment = 64 : i64} : memref<4x4xi8, #tsl.tsl<[2, 2] -> (10, 2), [2, 2] -> (4, 1)>, "L1">
// CHECK-NEXT:   "memref.copy"(%0, %1) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">, memref<4x4xi8, #tsl.tsl<[2, 2] -> (10, 2), [2, 2] -> (4, 1)>, "L1">) -> ()
// CHECK-NEXT:   "test.op"(%1) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (10, 2), [2, 2] -> (4, 1)>, "L1">) -> ()
// CHECK-NEXT:   "memref.copy"(%1, %0) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (10, 2), [2, 2] -> (4, 1)>, "L1">, memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>, "L1">) -> ()
// CHECK-NEXT: }

// -----

// a layout cast on a memref subview

"memref.global"() <{alignment = 64 : i64, constant, initial_value, sym_name = "global", sym_visibility = "private", type = memref<16x16xi8>}> : () -> ()
%0 = memref.get_global @global : memref<16x16xi8>
%1 = memref.subview %0[0, 0] [8, 16] [1, 1] : memref<16x16xi8> to memref<8x16xi8, strided<[16, 1], offset: 0>>
%2 = "snax.layout_cast"(%1) : (memref<8x16xi8, strided<[16, 1], offset: 0>>) -> memref<8x16xi8, #tsl.tsl<[8] -> (8), [2, 8] -> (64, 1)>>
"test.op"(%1) : (memref<4x4xi8, #tsl.tsl<[2, 2] -> (8, 2), [2, 2] -> (4, 1)>>) -> ()
