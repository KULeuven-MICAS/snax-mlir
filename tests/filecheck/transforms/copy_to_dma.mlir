// RUN: ./compiler/snax-opt --split-input-file %s -p snax-copy-to-dma | filecheck %s

"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<?xi32>, memref<?xi32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>):
    "memref.copy"(%arg0, %arg1) : (memref<?xi32>, memref<?xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func public @simple_mult(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>) {
// CHECK-NEXT:      %0 = arith.constant 0 : index
// CHECK-NEXT:      %1 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index
// CHECK-NEXT:      %2 = arith.constant 4 : index
// CHECK-NEXT:      %3 = arith.muli %1, %2 : index
// CHECK-NEXT:      %4 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
// CHECK-NEXT:      %5 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
// CHECK-NEXT:      func.call @snax_dma_1d_transfer(%4, %5, %3) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @snax_dma_1d_transfer(index, index, index) -> ()
// CHECK-NEXT:  }

// -----

"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<?x?xi32>, memref<?x?xi32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<?x?xi32>, %arg1 : memref<?x?xi32>):
    "memref.copy"(%arg0, %arg1) : (memref<?x?xi32>, memref<?x?xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:  builtin.module {
// CHECK-NEXT:    func.func public @simple_mult(%arg0 : memref<?x?xi32>, %arg1 : memref<?x?xi32>) {
// CHECK-NEXT:      %0 = arith.constant 0 : index
// CHECK-NEXT:      %1 = "memref.dim"(%arg0, %0) : (memref<?x?xi32>, index) -> index
// CHECK-NEXT:      %2 = arith.constant 1 : index
// CHECK-NEXT:      %3 = "memref.dim"(%arg0, %2) : (memref<?x?xi32>, index) -> index
// CHECK-NEXT:      %4 = arith.muli %1, %3 : index
// CHECK-NEXT:      %5 = arith.constant 4 : index
// CHECK-NEXT:      %6 = arith.muli %4, %5 : index
// CHECK-NEXT:      %7 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?x?xi32>) -> index
// CHECK-NEXT:      %8 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?x?xi32>) -> index
// CHECK-NEXT:      func.call @snax_dma_1d_transfer(%7, %8, %6) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @snax_dma_1d_transfer(index, index, index) -> ()
// CHECK-NEXT:  }

// -----

"builtin.module"() ({
    "func.func"() <{"sym_name" = "transform_copy", "function_type" = (memref<5xi32, strided<[1], offset:?>, "L3">, memref<5xi32, strided<[1], offset:?>, "L1">) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<5xi32, strided<[1], offset:?>, "L3">, %arg1 : memref<5xi32, strided<[1], offset:?>, "L1">):
    "memref.copy"(%arg0, %arg1) : (memref<5xi32, strided<[1], offset:?>, "L3">, memref<5xi32, strided<[1], offset:?>, "L1">) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:  builtin.module {
// CHECK-NEXT:    func.func public @transform_copy(%arg0 : memref<5xi32, strided<[1], offset: ?>, "L3">, %arg1 : memref<5xi32, strided<[1], offset: ?>, "L1">) {
// CHECK-NEXT:      %0 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<5xi32, strided<[1], offset: ?>, "L3">) -> index
// CHECK-NEXT:      %1 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<5xi32, strided<[1], offset: ?>, "L1">) -> index
// CHECK-NEXT:      %2, %3, %4, %5 = "memref.extract_strided_metadata"(%arg0) : (memref<5xi32, strided<[1], offset: ?>, "L3">) -> (memref<i32, "L3">, index, index, index)
// CHECK-NEXT:      %6 = arith.constant 4 : index
// CHECK-NEXT:      %7 = arith.muli %6, %3 : index
// CHECK-NEXT:      %8 = arith.addi %0, %7 : index
// CHECK-NEXT:      %9, %10, %11, %12 = "memref.extract_strided_metadata"(%arg1) : (memref<5xi32, strided<[1], offset: ?>, "L1">) -> (memref<i32, "L1">, index, index, index)
// CHECK-NEXT:      %13 = arith.constant 4 : index
// CHECK-NEXT:      %14 = arith.muli %13, %10 : index
// CHECK-NEXT:      %15 = arith.addi %1, %14 : index
// CHECK-NEXT:      %16 = arith.constant 0 : index
// CHECK-NEXT:      %17 = "memref.dim"(%arg0, %16) : (memref<5xi32, strided<[1], offset: ?>, "L3">, index) -> index
// CHECK-NEXT:      %18 = arith.constant 5 : index
// CHECK-NEXT:      %19, %20, %21, %22 = "memref.extract_strided_metadata"(%arg0) : (memref<5xi32, strided<[1], offset: ?>, "L3">) -> (memref<i32, "L3">, index, index, index)
// CHECK-NEXT:      %23 = arith.constant 4 : index
// CHECK-NEXT:      %24 = arith.constant 4 : index
// CHECK-NEXT:      %25 = arith.constant 4 : index
// CHECK-NEXT:      %26, %27, %28, %29 = "memref.extract_strided_metadata"(%arg1) : (memref<5xi32, strided<[1], offset: ?>, "L1">) -> (memref<i32, "L1">, index, index, index)
// CHECK-NEXT:      %30 = arith.constant 4 : index
// CHECK-NEXT:      %31 = arith.constant 4 : index
// CHECK-NEXT:      %32 = arith.constant 4 : index
// CHECK-NEXT:      %33 = arith.constant 0 : index
// CHECK-NEXT:      %34 = "memref.dim"(%arg0, %33) : (memref<5xi32, strided<[1], offset: ?>, "L3">, index) -> index
// CHECK-NEXT:      %35 = arith.constant 4 : index
// CHECK-NEXT:      %36 = arith.muli %34, %35 : index
// CHECK-NEXT:      func.call @snax_dma_1d_transfer(%8, %15, %36) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @snax_dma_1d_transfer(index, index, index) -> ()
// CHECK-NEXT:  }

// -----

"builtin.module"() ({
  "func.func"() <{"sym_name" = "transform_copy", "function_type" = (memref<?x?xi32, strided<[?, 1]>, "L3">, memref<?x?xi32, strided<[?, 1]>, "L1">) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<?x?xi32, strided<[?, 1]>, "L3">, %arg1 : memref<?x?xi32, strided<[?, 1]>, "L1">):
    "memref.copy"(%arg0, %arg1) : (memref<?x?xi32, strided<[?, 1]>, "L3">, memref<?x?xi32, strided<[?, 1]>, "L1">) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:  builtin.module {
// CHECK-NEXT:    func.func public @transform_copy(%arg0 : memref<?x?xi32, strided<[?, 1]>, "L3">, %arg1 : memref<?x?xi32, strided<[?, 1]>, "L1">) {
// CHECK-NEXT:      %0 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?x?xi32, strided<[?, 1]>, "L3">) -> index
// CHECK-NEXT:      %1 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?x?xi32, strided<[?, 1]>, "L1">) -> index
// CHECK-NEXT:      %2 = arith.constant 0 : index
// CHECK-NEXT:      %3 = "memref.dim"(%arg0, %2) : (memref<?x?xi32, strided<[?, 1]>, "L3">, index) -> index
// CHECK-NEXT:      %4 = arith.constant 1 : index
// CHECK-NEXT:      %5 = "memref.dim"(%arg0, %4) : (memref<?x?xi32, strided<[?, 1]>, "L3">, index) -> index
// CHECK-NEXT:      %6 = arith.constant 1 : index
// CHECK-NEXT:      %7 = arith.divui %3, %6 : index
// CHECK-NEXT:      %8 = arith.constant 1 : index
// CHECK-NEXT:      %9 = arith.divui %5, %8 : index
// CHECK-NEXT:      %10, %11, %12, %13, %14, %15 = "memref.extract_strided_metadata"(%arg0) : (memref<?x?xi32, strided<[?, 1]>, "L3">) -> (memref<i32, "L3">, index, index, index, index, index)
// CHECK-NEXT:      %16 = arith.constant 4 : index
// CHECK-NEXT:      %17 = arith.constant 4 : index
// CHECK-NEXT:      %18 = arith.constant 4 : index
// CHECK-NEXT:      %19 = arith.muli %14, %16 : index
// CHECK-NEXT:      %20, %21, %22, %23, %24, %25 = "memref.extract_strided_metadata"(%arg1) : (memref<?x?xi32, strided<[?, 1]>, "L1">) -> (memref<i32, "L1">, index, index, index, index, index)
// CHECK-NEXT:      %26 = arith.constant 4 : index
// CHECK-NEXT:      %27 = arith.constant 4 : index
// CHECK-NEXT:      %28 = arith.constant 4 : index
// CHECK-NEXT:      %29 = arith.muli %24, %26 : index
// CHECK-NEXT:      %30 = arith.constant 0 : index
// CHECK-NEXT:      %31 = "memref.dim"(%arg0, %30) : (memref<?x?xi32, strided<[?, 1]>, "L3">, index) -> index
// CHECK-NEXT:      %32 = arith.constant 1 : index
// CHECK-NEXT:      %33 = "memref.dim"(%arg0, %32) : (memref<?x?xi32, strided<[?, 1]>, "L3">, index) -> index
// CHECK-NEXT:      %34 = arith.muli %31, %33 : index
// CHECK-NEXT:      %35 = arith.constant 4 : index
// CHECK-NEXT:      %36 = arith.muli %34, %35 : index
// CHECK-NEXT:      func.call @snax_dma_1d_transfer(%0, %1, %36) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @snax_dma_1d_transfer(index, index, index) -> ()
// CHECK-NEXT:  }

// -----

"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<5x5xi32, strided<[10, 1]>>, memref<5x5xi32, strided<[20, 1]>>) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<5x5xi32, strided<[10, 1]>>, %arg1 : memref<5x5xi32, strided<[20, 1]>>):
    "memref.copy"(%arg0, %arg1) : (memref<5x5xi32, strided<[10, 1]>>, memref<5x5xi32, strided<[20, 1]>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:  builtin.module {
// CHECK-NEXT:    func.func public @simple_mult(%arg0 : memref<5x5xi32, strided<[10, 1]>>, %arg1 : memref<5x5xi32, strided<[20, 1]>>) {
// CHECK-NEXT:      %0 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<5x5xi32, strided<[10, 1]>>) -> index
// CHECK-NEXT:      %1 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<5x5xi32, strided<[20, 1]>>) -> index
// CHECK-NEXT:      %2 = arith.constant 0 : index
// CHECK-NEXT:      %3 = "memref.dim"(%arg0, %2) : (memref<5x5xi32, strided<[10, 1]>>, index) -> index
// CHECK-NEXT:      %4 = arith.constant 1 : index
// CHECK-NEXT:      %5 = "memref.dim"(%arg0, %4) : (memref<5x5xi32, strided<[10, 1]>>, index) -> index
// CHECK-NEXT:      %6 = arith.constant 5 : index
// CHECK-NEXT:      %7 = arith.constant 5 : index
// CHECK-NEXT:      %8, %9, %10, %11, %12, %13 = "memref.extract_strided_metadata"(%arg0) : (memref<5x5xi32, strided<[10, 1]>>) -> (memref<i32>, index, index, index, index, index)
// CHECK-NEXT:      %14 = arith.constant 4 : index
// CHECK-NEXT:      %15 = arith.constant 40 : index
// CHECK-NEXT:      %16 = arith.constant 4 : index
// CHECK-NEXT:      %17 = arith.constant 40 : index
// CHECK-NEXT:      %18, %19, %20, %21, %22, %23 = "memref.extract_strided_metadata"(%arg1) : (memref<5x5xi32, strided<[20, 1]>>) -> (memref<i32>, index, index, index, index, index)
// CHECK-NEXT:      %24 = arith.constant 4 : index
// CHECK-NEXT:      %25 = arith.constant 80 : index
// CHECK-NEXT:      %26 = arith.constant 4 : index
// CHECK-NEXT:      %27 = arith.constant 80 : index
// CHECK-NEXT:      %28 = arith.constant 20 : index
// CHECK-NEXT:      func.call @snax_dma_2d_transfer(%0, %1, %28, %17, %27, %6) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @snax_dma_2d_transfer(index, index, index, index, index, index) -> ()
// CHECK-NEXT:  }

// -----

"builtin.module"() ({
  "func.func"() <{"sym_name" = "transform_copy", "function_type" = (memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">, memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 1), [2, 4] -> (32, 4)>, "L1">) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">, %arg1 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 1), [2, 4] -> (32, 4)>, "L1">):
    "memref.copy"(%arg0, %arg1) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">, memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 1), [2, 4] -> (32, 4)>, "L1">) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:  builtin.module {
// CHECK-NEXT:    func.func public @transform_copy(%arg0 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">, %arg1 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 1), [2, 4] -> (32, 4)>, "L1">) {
// CHECK-NEXT:      %0 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">) -> index
// CHECK-NEXT:      %1 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 1), [2, 4] -> (32, 4)>, "L1">) -> index
// CHECK-NEXT:      %2 = arith.constant 0 : index
// CHECK-NEXT:      %3 = "memref.dim"(%arg0, %2) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">, index) -> index
// CHECK-NEXT:      %4 = arith.constant 1 : index
// CHECK-NEXT:      %5 = "memref.dim"(%arg0, %4) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">, index) -> index
// CHECK-NEXT:      %6 = arith.constant 2 : index
// CHECK-NEXT:      %7 = arith.constant 4 : index
// CHECK-NEXT:      %8 = arith.constant 2 : index
// CHECK-NEXT:      %9 = arith.constant 4 : index
// CHECK-NEXT:      %10 = arith.constant 128 : index
// CHECK-NEXT:      %11 = arith.constant 32 : index
// CHECK-NEXT:      %12 = arith.constant 128 : index
// CHECK-NEXT:      %13 = arith.constant 4 : index
// CHECK-NEXT:      %14 = arith.constant 16 : index
// CHECK-NEXT:      %15 = arith.constant 128 : index
// CHECK-NEXT:      %16 = arith.constant 16 : index
// CHECK-NEXT:      %17 = arith.constant 128 : index
// CHECK-NEXT:      %18 = arith.constant 4 : index
// CHECK-NEXT:      %19 = arith.constant 64 : index
// CHECK-NEXT:      %20 = arith.constant 16 : index
// CHECK-NEXT:      %21 = arith.constant 0 : index
// CHECK-NEXT:      %22 = arith.constant 1 : index
// CHECK-NEXT:      scf.for %23 = %21 to %6 step %22 {
// CHECK-NEXT:        %24 = arith.muli %23, %14 : index
// CHECK-NEXT:        %25 = arith.addi %0, %24 : index
// CHECK-NEXT:        %26 = arith.muli %23, %19 : index
// CHECK-NEXT:        %27 = arith.addi %1, %26 : index
// CHECK-NEXT:        scf.for %28 = %21 to %8 step %22 {
// CHECK-NEXT:          %29 = arith.muli %28, %12 : index
// CHECK-NEXT:          %30 = arith.addi %25, %29 : index
// CHECK-NEXT:          %31 = arith.muli %28, %17 : index
// CHECK-NEXT:          %32 = arith.addi %27, %31 : index
// CHECK-NEXT:          func.call @snax_dma_2d_transfer(%30, %32, %20, %11, %16, %9) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @snax_dma_2d_transfer(index, index, index, index, index, index) -> ()
// CHECK-NEXT:  }

