// RUN: snax-opt --split-input-file -p convert-memref-to-arith %s | filecheck %s

%0 = "test.op"() : () -> (memref<16x16xi8, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>>)
%1 = "test.op"() : () -> (index)
%2 = "test.op"() : () -> (index)
%3 = memref.subview %0[%1, 0] [8, 16] [1, 1] : memref<16x16xi8, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>> to memref<8x16xi8, #tsl.tsl<[8] -> (8), [2, 8] -> (64, 1)>>
%4 = "memref.extract_aligned_pointer_as_index"(%3) : (memref<8x16xi8, #tsl.tsl<[8] -> (8), [2, 8] -> (64, 1)>>) -> index
"test.op"(%4) : (index) -> ()

%5 = memref.subview %0[%1, %2] [8, 16] [1, 1] : memref<16x16xi8, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>> to memref<8x8xi8, #tsl.tsl<[8] -> (8), [8] -> (1)>>
%6 = "memref.extract_aligned_pointer_as_index"(%5) : (memref<8x8xi8, #tsl.tsl<[8] -> (8), [8] -> (1)>>) -> index
"test.op"(%6) : (index) -> ()

// CHECK: builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> memref<16x16xi8, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>>
// CHECK-NEXT:   %1 = "test.op"() : () -> index
// CHECK-NEXT:   %2 = "test.op"() : () -> index
// CHECK-NEXT:   %3 = memref.subview %0[%1, 0] [8, 16] [1, 1] : memref<16x16xi8, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>> to memref<8x16xi8, #tsl.tsl<[8] -> (8), [2, 8] -> (64, 1)>>
// CHECK-NEXT:   %4 = "memref.extract_aligned_pointer_as_index"(%0) : (memref<16x16xi8, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>>) -> index
// CHECK-NEXT:   %5 = arith.constant 1 : index
// CHECK-NEXT:   %6 = arith.constant 128 : index
// CHECK-NEXT:   %7 = arith.muli %6, %5 : index
// CHECK-NEXT:   %8 = arith.constant 8 : index
// CHECK-NEXT:   %9 = arith.divui %1, %8 : index
// CHECK-NEXT:   %10 = arith.muli %9, %7 : index
// CHECK-NEXT:   %11 = arith.addi %4, %10 : index
// CHECK-NEXT:   "test.op"(%11) : (index) -> ()
// CHECK-NEXT:   %12 = memref.subview %0[%1, %2] [8, 16] [1, 1] : memref<16x16xi8, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>> to memref<8x8xi8, #tsl.tsl<[8] -> (8), [8] -> (1)>>
// CHECK-NEXT:   %13 = "memref.extract_aligned_pointer_as_index"(%0) : (memref<16x16xi8, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>>) -> index
// CHECK-NEXT:   %14 = arith.constant 1 : index
// CHECK-NEXT:   %15 = arith.constant 128 : index
// CHECK-NEXT:   %16 = arith.muli %15, %14 : index
// CHECK-NEXT:   %17 = arith.constant 8 : index
// CHECK-NEXT:   %18 = arith.divui %1, %17 : index
// CHECK-NEXT:   %19 = arith.muli %18, %16 : index
// CHECK-NEXT:   %20 = arith.addi %13, %19 : index
// CHECK-NEXT:   %21 = arith.constant 64 : index
// CHECK-NEXT:   %22 = arith.muli %21, %14 : index
// CHECK-NEXT:   %23 = arith.constant 8 : index
// CHECK-NEXT:   %24 = arith.divui %2, %23 : index
// CHECK-NEXT:   %25 = arith.muli %24, %22 : index
// CHECK-NEXT:   %26 = arith.addi %20, %25 : index
// CHECK-NEXT:   "test.op"(%26) : (index) -> ()
// CHECK-NEXT: }
