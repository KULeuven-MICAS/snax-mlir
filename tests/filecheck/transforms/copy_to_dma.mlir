// RUN: snax-opt --split-input-file %s -p snax-copy-to-dma | filecheck %s

"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<?xi32>, memref<?xi32>) -> (), "sym_visibility" = "public"}> ({
  ^bb0(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>):
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
  ^bb0(%arg0 : memref<?x?xi32>, %arg1 : memref<?x?xi32>):
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
  ^bb0(%arg0 : memref<5xi32, strided<[1], offset:?>, "L3">, %arg1 : memref<5xi32, strided<[1], offset:?>, "L1">):
    "memref.copy"(%arg0, %arg1) : (memref<5xi32, strided<[1], offset:?>, "L3">, memref<5xi32, strided<[1], offset:?>, "L1">) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func public @transform_copy(%arg0 : memref<5xi32, strided<[1], offset: ?>, "L3">, %arg1 : memref<5xi32, strided<[1], offset: ?>, "L1">) {
// CHECK-NEXT:     %0 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<5xi32, strided<[1], offset: ?>, "L3">) -> index
// CHECK-NEXT:     %1 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<5xi32, strided<[1], offset: ?>, "L1">) -> index
// CHECK-NEXT:     %2, %3, %4, %5 = memref.extract_strided_metadata %arg0 : memref<5xi32, strided<[1], offset: ?>, "L3"> -> memref<i32, "L3">, index, index, index
// CHECK-NEXT:     %6 = arith.constant 4 : index
// CHECK-NEXT:     %7 = arith.muli %6, %3 : index
// CHECK-NEXT:     %8 = arith.addi %0, %7 : index
// CHECK-NEXT:     %9, %10, %11, %12 = memref.extract_strided_metadata %arg1 : memref<5xi32, strided<[1], offset: ?>, "L1"> -> memref<i32, "L1">, index, index, index
// CHECK-NEXT:     %13 = arith.constant 4 : index
// CHECK-NEXT:     %14 = arith.muli %13, %10 : index
// CHECK-NEXT:     %15 = arith.addi %1, %14 : index
// CHECK-NEXT:     %16 = arith.constant 0 : index
// CHECK-NEXT:     %17 = "memref.dim"(%arg0, %16) : (memref<5xi32, strided<[1], offset: ?>, "L3">, index) -> index
// CHECK-NEXT:     %18 = arith.constant 5 : index
// CHECK-NEXT:     %19, %20, %21, %22 = memref.extract_strided_metadata %arg0 : memref<5xi32, strided<[1], offset: ?>, "L3"> -> memref<i32, "L3">, index, index, index
// CHECK-NEXT:     %23 = arith.constant 4 : index
// CHECK-NEXT:     %24 = arith.constant 4 : index
// CHECK-NEXT:     %25 = arith.muli %18, %24 : index
// CHECK-NEXT:     %26 = arith.constant 4 : index
// CHECK-NEXT:     %27, %28, %29, %30 = memref.extract_strided_metadata %arg1 : memref<5xi32, strided<[1], offset: ?>, "L1"> -> memref<i32, "L1">, index, index, index
// CHECK-NEXT:     %31 = arith.constant 4 : index
// CHECK-NEXT:     %32 = arith.constant 4 : index
// CHECK-NEXT:     %33 = arith.muli %18, %32 : index
// CHECK-NEXT:     %34 = arith.constant 4 : index
// CHECK-NEXT:     %35 = arith.constant 0 : index
// CHECK-NEXT:     %36 = "memref.dim"(%arg0, %35) : (memref<5xi32, strided<[1], offset: ?>, "L3">, index) -> index
// CHECK-NEXT:     %37 = arith.constant 4 : index
// CHECK-NEXT:     %38 = arith.muli %36, %37 : index
// CHECK-NEXT:     func.call @snax_dma_1d_transfer(%8, %15, %38) : (index, index, index) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @snax_dma_1d_transfer(index, index, index) -> ()
// CHECK-NEXT: }

// -----

"builtin.module"() ({
  "func.func"() <{"sym_name" = "transform_copy", "function_type" = (memref<?x?xi32, strided<[?, 1]>, "L3">, memref<?x?xi32, strided<[?, 1]>, "L1">) -> (), "sym_visibility" = "public"}> ({
  ^bb0(%arg0 : memref<?x?xi32, strided<[?, 1]>, "L3">, %arg1 : memref<?x?xi32, strided<[?, 1]>, "L1">):
    "memref.copy"(%arg0, %arg1) : (memref<?x?xi32, strided<[?, 1]>, "L3">, memref<?x?xi32, strided<[?, 1]>, "L1">) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func public @transform_copy(%arg0 : memref<?x?xi32, strided<[?, 1]>, "L3">, %arg1 : memref<?x?xi32, strided<[?, 1]>, "L1">) {
// CHECK-NEXT:     %0 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?x?xi32, strided<[?, 1]>, "L3">) -> index
// CHECK-NEXT:     %1 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?x?xi32, strided<[?, 1]>, "L1">) -> index
// CHECK-NEXT:     %2 = arith.constant 0 : index
// CHECK-NEXT:     %3 = "memref.dim"(%arg0, %2) : (memref<?x?xi32, strided<[?, 1]>, "L3">, index) -> index
// CHECK-NEXT:     %4 = arith.constant 1 : index
// CHECK-NEXT:     %5 = "memref.dim"(%arg0, %4) : (memref<?x?xi32, strided<[?, 1]>, "L3">, index) -> index
// CHECK-NEXT:     %6 = arith.constant 1 : index
// CHECK-NEXT:     %7 = arith.divui %3, %6 : index
// CHECK-NEXT:     %8 = arith.constant 1 : index
// CHECK-NEXT:     %9 = arith.divui %5, %8 : index
// CHECK-NEXT:     %10, %11, %12, %13, %14, %15 = memref.extract_strided_metadata %arg0 : memref<?x?xi32, strided<[?, 1]>, "L3"> -> memref<i32, "L3">, index, index, index, index, index
// CHECK-NEXT:     %16 = arith.constant 4 : index
// CHECK-NEXT:     %17 = arith.muli %14, %16 : index
// CHECK-NEXT:     %18 = arith.constant 4 : index
// CHECK-NEXT:     %19 = arith.muli %9, %18 : index
// CHECK-NEXT:     %20 = arith.constant 4 : index
// CHECK-NEXT:     %21 = arith.muli %17, %7 : index
// CHECK-NEXT:     %22, %23, %24, %25, %26, %27 = memref.extract_strided_metadata %arg1 : memref<?x?xi32, strided<[?, 1]>, "L1"> -> memref<i32, "L1">, index, index, index, index, index
// CHECK-NEXT:     %28 = arith.constant 4 : index
// CHECK-NEXT:     %29 = arith.muli %26, %28 : index
// CHECK-NEXT:     %30 = arith.constant 4 : index
// CHECK-NEXT:     %31 = arith.muli %9, %30 : index
// CHECK-NEXT:     %32 = arith.constant 4 : index
// CHECK-NEXT:     %33 = arith.muli %29, %7 : index
// CHECK-NEXT:     %34 = arith.constant 0 : index
// CHECK-NEXT:     %35 = "memref.dim"(%arg0, %34) : (memref<?x?xi32, strided<[?, 1]>, "L3">, index) -> index
// CHECK-NEXT:     %36 = arith.constant 1 : index
// CHECK-NEXT:     %37 = "memref.dim"(%arg0, %36) : (memref<?x?xi32, strided<[?, 1]>, "L3">, index) -> index
// CHECK-NEXT:     %38 = arith.muli %35, %37 : index
// CHECK-NEXT:     %39 = arith.constant 4 : index
// CHECK-NEXT:     %40 = arith.muli %38, %39 : index
// CHECK-NEXT:     func.call @snax_dma_1d_transfer(%0, %1, %40) : (index, index, index) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @snax_dma_1d_transfer(index, index, index) -> ()
// CHECK-NEXT: }

// -----

"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<5x5xi32, strided<[10, 1]>>, memref<5x5xi32, strided<[20, 1]>>) -> (), "sym_visibility" = "public"}> ({
  ^bb0(%arg0 : memref<5x5xi32, strided<[10, 1]>>, %arg1 : memref<5x5xi32, strided<[20, 1]>>):
    "memref.copy"(%arg0, %arg1) : (memref<5x5xi32, strided<[10, 1]>>, memref<5x5xi32, strided<[20, 1]>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%arg0 : memref<5x5xi32, strided<[10, 1]>>, %arg1 : memref<5x5xi32, strided<[20, 1]>>) {
// CHECK-NEXT:     %0 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<5x5xi32, strided<[10, 1]>>) -> index
// CHECK-NEXT:     %1 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<5x5xi32, strided<[20, 1]>>) -> index
// CHECK-NEXT:     %2 = arith.constant 0 : index
// CHECK-NEXT:     %3 = "memref.dim"(%arg0, %2) : (memref<5x5xi32, strided<[10, 1]>>, index) -> index
// CHECK-NEXT:     %4 = arith.constant 1 : index
// CHECK-NEXT:     %5 = "memref.dim"(%arg0, %4) : (memref<5x5xi32, strided<[10, 1]>>, index) -> index
// CHECK-NEXT:     %6 = arith.constant 5 : index
// CHECK-NEXT:     %7 = arith.constant 5 : index
// CHECK-NEXT:     %8, %9, %10, %11, %12, %13 = memref.extract_strided_metadata %arg0 : memref<5x5xi32, strided<[10, 1]>> -> memref<i32>, index, index, index, index, index
// CHECK-NEXT:     %14 = arith.constant 4 : index
// CHECK-NEXT:     %15 = arith.constant 40 : index
// CHECK-NEXT:     %16 = arith.muli %6, %15 : index
// CHECK-NEXT:     %17 = arith.constant 4 : index
// CHECK-NEXT:     %18 = arith.constant 40 : index
// CHECK-NEXT:     %19, %20, %21, %22, %23, %24 = memref.extract_strided_metadata %arg1 : memref<5x5xi32, strided<[20, 1]>> -> memref<i32>, index, index, index, index, index
// CHECK-NEXT:     %25 = arith.constant 4 : index
// CHECK-NEXT:     %26 = arith.constant 80 : index
// CHECK-NEXT:     %27 = arith.muli %6, %26 : index
// CHECK-NEXT:     %28 = arith.constant 4 : index
// CHECK-NEXT:     %29 = arith.constant 80 : index
// CHECK-NEXT:     %30 = arith.constant 20 : index
// CHECK-NEXT:     func.call @snax_dma_2d_transfer(%0, %1, %30, %18, %29, %6) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @snax_dma_2d_transfer(index, index, index, index, index, index) -> ()
// CHECK-NEXT: }

// -----

"builtin.module"() ({
  "func.func"() <{"sym_name" = "transform_copy", "function_type" = (memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">, memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 1), [2, 4] -> (32, 4)>, "L1">) -> (), "sym_visibility" = "public"}> ({
  ^bb0(%arg0 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">, %arg1 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 1), [2, 4] -> (32, 4)>, "L1">):
    "memref.copy"(%arg0, %arg1) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">, memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 1), [2, 4] -> (32, 4)>, "L1">) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func public @transform_copy(%arg0 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">, %arg1 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 1), [2, 4] -> (32, 4)>, "L1">) {
// CHECK-NEXT:     %0 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">) -> index
// CHECK-NEXT:     %1 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 1), [2, 4] -> (32, 4)>, "L1">) -> index
// CHECK-NEXT:     %2 = arith.constant 0 : index
// CHECK-NEXT:     %3 = "memref.dim"(%arg0, %2) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">, index) -> index
// CHECK-NEXT:     %4 = arith.constant 1 : index
// CHECK-NEXT:     %5 = "memref.dim"(%arg0, %4) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">, index) -> index
// CHECK-NEXT:     %6 = arith.constant 2 : index
// CHECK-NEXT:     %7 = arith.constant 4 : index
// CHECK-NEXT:     %8 = arith.constant 2 : index
// CHECK-NEXT:     %9 = arith.constant 4 : index
// CHECK-NEXT:     %10 = arith.constant 128 : index
// CHECK-NEXT:     %11 = arith.muli %8, %10 : index
// CHECK-NEXT:     %12 = arith.constant 32 : index
// CHECK-NEXT:     %13 = arith.constant 128 : index
// CHECK-NEXT:     %14 = arith.constant 4 : index
// CHECK-NEXT:     %15 = arith.constant 16 : index
// CHECK-NEXT:     %16 = arith.constant 128 : index
// CHECK-NEXT:     %17 = arith.muli %8, %16 : index
// CHECK-NEXT:     %18 = arith.constant 16 : index
// CHECK-NEXT:     %19 = arith.constant 128 : index
// CHECK-NEXT:     %20 = arith.constant 4 : index
// CHECK-NEXT:     %21 = arith.constant 64 : index
// CHECK-NEXT:     %22 = arith.constant 16 : index
// CHECK-NEXT:     %23 = arith.constant 0 : index
// CHECK-NEXT:     %24 = arith.constant 1 : index
// CHECK-NEXT:     scf.for %25 = %23 to %6 step %24 {
// CHECK-NEXT:       %26 = arith.muli %25, %15 : index
// CHECK-NEXT:       %27 = arith.addi %0, %26 : index
// CHECK-NEXT:       %28 = arith.muli %25, %21 : index
// CHECK-NEXT:       %29 = arith.addi %1, %28 : index
// CHECK-NEXT:       scf.for %30 = %23 to %8 step %24 {
// CHECK-NEXT:         %31 = arith.muli %30, %13 : index
// CHECK-NEXT:         %32 = arith.addi %27, %31 : index
// CHECK-NEXT:         %33 = arith.muli %30, %19 : index
// CHECK-NEXT:         %34 = arith.addi %29, %33 : index
// CHECK-NEXT:         func.call @snax_dma_2d_transfer(%32, %34, %22, %12, %18, %9) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @snax_dma_2d_transfer(index, index, index, index, index, index) -> ()
// CHECK-NEXT: }
