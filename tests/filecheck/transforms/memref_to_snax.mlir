// RUN: snax-opt --split-input-file %s -p memref-to-snax | filecheck %s

"builtin.module"() ({
  %0 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<16x16xi32>
}) : () -> ()

// expect nothing to change because no memory space is specified
// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi32>
// CHECK-NEXT:  }


// -----

"builtin.module"() ({
  %0 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<16x16xi32, "L1">
}) : () -> ()

// CHECK:  builtin.module {
// CHECK-NEXT:    %0 = arith.constant 16 : index
// CHECK-NEXT:    %1 = arith.constant 16 : index
// CHECK-NEXT:    %2 = arith.constant 4 : index
// CHECK-NEXT:    %3 = arith.muli %0, %2 : index
// CHECK-NEXT:    %4 = arith.muli %1, %3 : index
// CHECK-NEXT:    %5 = "snax.alloc"(%4, %0, %1) <{memory_space = "L1", alignment = 64 : i64}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:    %6 = builtin.unrealized_conversion_cast %5 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)> to memref<16x16xi32, "L1">
// CHECK-NEXT:  }

// -----

"builtin.module"() ({
  %0 = "test.op"() : () -> (index)
  %1 = "memref.alloc"(%0) <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 1, 0>}> : (index) -> memref<?x16xi32, "L1">
}) : () -> ()

// CHECK:  builtin.module {
// CHECK-NEXT:    %0 = "test.op"() : () -> index
// CHECK-NEXT:    %1 = arith.constant 16 : index
// CHECK-NEXT:    %2 = arith.constant 4 : index
// CHECK-NEXT:    %3 = arith.muli %0, %2 : index
// CHECK-NEXT:    %4 = arith.muli %1, %3 : index
// CHECK-NEXT:    %5 = "snax.alloc"(%4, %0, %1) <{memory_space = "L1", alignment = 64 : i64}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:    %6 = builtin.unrealized_conversion_cast %5 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)> to memref<?x16xi32, "L1">
// CHECK-NEXT:  }

// -----

"builtin.module"() ({
  %0 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, "L1">
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = arith.constant 8 : index
// CHECK-NEXT:   %1 = arith.constant 8 : index
// CHECK-NEXT:   %2 = arith.constant 2 : index
// CHECK-NEXT:   %3 = arith.constant 4 : index
// CHECK-NEXT:   %4 = arith.constant 2 : index
// CHECK-NEXT:   %5 = arith.constant 4 : index
// CHECK-NEXT:   %6 = arith.constant 512 : index
// CHECK-NEXT:   %7 = arith.muli %4, %6 : index
// CHECK-NEXT:   %8 = arith.constant 128 : index
// CHECK-NEXT:   %9 = arith.constant 512 : index
// CHECK-NEXT:   %10 = arith.constant 16 : index
// CHECK-NEXT:   %11 = arith.constant 64 : index
// CHECK-NEXT:   %12 = arith.constant 1 : index
// CHECK-NEXT:   %13 = arith.constant 0 : index
// CHECK-NEXT:   %14 = arith.subi %2, %12 : index
// CHECK-NEXT:   %15 = arith.muli %14, %11 : index
// CHECK-NEXT:   %16 = arith.addi %13, %15 : index
// CHECK-NEXT:   %17 = arith.subi %3, %12 : index
// CHECK-NEXT:   %18 = arith.muli %17, %10 : index
// CHECK-NEXT:   %19 = arith.addi %16, %18 : index
// CHECK-NEXT:   %20 = arith.subi %4, %12 : index
// CHECK-NEXT:   %21 = arith.muli %20, %9 : index
// CHECK-NEXT:   %22 = arith.addi %19, %21 : index
// CHECK-NEXT:   %23 = arith.subi %5, %12 : index
// CHECK-NEXT:   %24 = arith.muli %23, %8 : index
// CHECK-NEXT:   %25 = arith.addi %22, %24 : index
// CHECK-NEXT:   %26 = arith.constant 4 : index
// CHECK-NEXT:   %27 = arith.addi %25, %26 : index
// CHECK-NEXT:   %28 = arith.muli %12, %27 : index
// CHECK-NEXT:   %29 = arith.constant 0 : index
// CHECK-NEXT:   %30 = arith.muli %29, %26 : index
// CHECK-NEXT:   %31 = arith.addi %28, %30 : index
// CHECK-NEXT:   %32 = "snax.alloc"(%31, %0, %1) <{memory_space = "L1", alignment = 64 : i64}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %33 = builtin.unrealized_conversion_cast %32 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)> to memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, "L1">
// CHECK-NEXT: }

// -----

"builtin.module"() ({
  %0 = "test.op"() : () -> (index)
  %1 = "memref.alloc"(%0) <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 1, 0>}> : (index) -> memref<8x?xi32, #tsl.tsl<[2, 4] -> (16, 4), [?, 4] -> (?, 32)>, "L1">
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> index
// CHECK-NEXT:   %1 = arith.constant 8 : index
// CHECK-NEXT:   %2 = arith.constant 2 : index
// CHECK-NEXT:   %3 = arith.constant 4 : index
// CHECK-NEXT:   %4 = arith.constant 4 : index
// CHECK-NEXT:   %5 = arith.divui %0, %4 : index
// CHECK-NEXT:   %6 = arith.constant 4 : index
// CHECK-NEXT:   %7 = arith.constant 128 : index
// CHECK-NEXT:   %8 = arith.muli %6, %7 : index
// CHECK-NEXT:   %9 = arith.constant 128 : index
// CHECK-NEXT:   %10 = arith.muli %8, %5 : index
// CHECK-NEXT:   %11 = arith.constant 16 : index
// CHECK-NEXT:   %12 = arith.constant 64 : index
// CHECK-NEXT:   %13 = arith.constant 1 : index
// CHECK-NEXT:   %14 = arith.constant 0 : index
// CHECK-NEXT:   %15 = arith.subi %2, %13 : index
// CHECK-NEXT:   %16 = arith.muli %15, %12 : index
// CHECK-NEXT:   %17 = arith.addi %14, %16 : index
// CHECK-NEXT:   %18 = arith.subi %3, %13 : index
// CHECK-NEXT:   %19 = arith.muli %18, %11 : index
// CHECK-NEXT:   %20 = arith.addi %17, %19 : index
// CHECK-NEXT:   %21 = arith.subi %5, %13 : index
// CHECK-NEXT:   %22 = arith.muli %21, %8 : index
// CHECK-NEXT:   %23 = arith.addi %20, %22 : index
// CHECK-NEXT:   %24 = arith.subi %6, %13 : index
// CHECK-NEXT:   %25 = arith.muli %24, %9 : index
// CHECK-NEXT:   %26 = arith.addi %23, %25 : index
// CHECK-NEXT:   %27 = arith.constant 4 : index
// CHECK-NEXT:   %28 = arith.addi %26, %27 : index
// CHECK-NEXT:   %29 = arith.muli %13, %28 : index
// CHECK-NEXT:   %30 = arith.constant 0 : index
// CHECK-NEXT:   %31 = arith.muli %30, %27 : index
// CHECK-NEXT:   %32 = arith.addi %29, %31 : index
// CHECK-NEXT:   %33 = "snax.alloc"(%32, %1, %0) <{memory_space = "L1", alignment = 64 : i64}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %34 = builtin.unrealized_conversion_cast %33 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)> to memref<8x?xi32, #tsl.tsl<[2, 4] -> (16, 4), [?, 4] -> (?, 32)>, "L1">
// CHECK-NEXT: }
