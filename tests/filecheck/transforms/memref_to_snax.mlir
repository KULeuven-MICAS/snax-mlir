// RUN: ./compiler/snax-opt --split-input-file %s -p memref-to-snax --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "memref.alloc"() <{"alignment" = 64 : i32, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<16x16xi32>
}) : () -> ()

// expect nothing to change because no memory space is specified
// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "memref.alloc"() <{"alignment" = 64 : i32, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<16x16xi32>
// CHECK-NEXT: }) : () -> ()


// -----

"builtin.module"() ({
  %0 = "memref.alloc"() <{"alignment" = 64 : i32, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<16x16xi32, 1: i32>
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "arith.constant"() <{"value" = 16 : index}> : () -> index
// CHECK-NEXT:   %1 = "arith.constant"() <{"value" = 16 : index}> : () -> index
// CHECK-NEXT:   %2 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:   %3 = "arith.muli"(%0, %2) : (index, index) -> index
// CHECK-NEXT:   %4 = "arith.muli"(%1, %3) : (index, index) -> index
// CHECK-NEXT:   %5 = "snax.alloc"(%4) <{"memory_space" = 1 : i32, "alignment" = 64 : i32}> : (index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %6 = "builtin.unrealized_conversion_cast"(%5) : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>) -> memref<16x16xi32, 1 : i32>
// CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  %0 = "test.op"() : () -> (index)
  %1 = "memref.alloc"(%0) <{"alignment" = 64 : i32, "operandSegmentSizes" = array<i32: 1, 0>}> : (index) -> memref<?x16xi32, 1: i32>
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "test.op"() : () -> index
// CHECK-NEXT:   %1 = "arith.constant"() <{"value" = 16 : index}> : () -> index
// CHECK-NEXT:   %2 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:   %3 = "arith.muli"(%0, %2) : (index, index) -> index
// CHECK-NEXT:   %4 = "arith.muli"(%1, %3) : (index, index) -> index
// CHECK-NEXT:   %5 = "snax.alloc"(%4) <{"memory_space" = 1 : i32, "alignment" = 64 : i32}> : (index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %6 = "builtin.unrealized_conversion_cast"(%5) : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>) -> memref<?x16xi32, 1 : i32>
// CHECK-NEXT: }) : () -> ()


// -----

"builtin.module"() ({
  %0 = "memref.alloc"() <{"alignment" = 64 : i32, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 1 : i32>
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "arith.constant"() <{"value" = 8 : index}> : () -> index
// CHECK-NEXT:   %1 = "arith.constant"() <{"value" = 8 : index}> : () -> index
// CHECK-NEXT:   %2 = "arith.constant"() <{"value" = 2 : index}> : () -> index
// CHECK-NEXT:   %3 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:   %4 = "arith.constant"() <{"value" = 2 : index}> : () -> index
// CHECK-NEXT:   %5 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:   %6 = "arith.constant"() <{"value" = 128 : index}> : () -> index
// CHECK-NEXT:   %7 = "arith.constant"() <{"value" = 32 : index}> : () -> index
// CHECK-NEXT:   %8 = "arith.constant"() <{"value" = 128 : index}> : () -> index
// CHECK-NEXT:   %9 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:   %10 = "arith.constant"() <{"value" = 16 : index}> : () -> index
// CHECK-NEXT:   %11 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:   %12 = "arith.constant"() <{"value" = 0 : index}> : () -> index
// CHECK-NEXT:   %13 = "arith.muli"(%2, %10) : (index, index) -> index
// CHECK-NEXT:   %14 = "arith.addi"(%12, %13) : (index, index) -> index
// CHECK-NEXT:   %15 = "arith.muli"(%3, %9) : (index, index) -> index
// CHECK-NEXT:   %16 = "arith.addi"(%14, %15) : (index, index) -> index
// CHECK-NEXT:   %17 = "arith.muli"(%4, %8) : (index, index) -> index
// CHECK-NEXT:   %18 = "arith.addi"(%16, %17) : (index, index) -> index
// CHECK-NEXT:   %19 = "arith.muli"(%5, %7) : (index, index) -> index
// CHECK-NEXT:   %20 = "arith.addi"(%18, %19) : (index, index) -> index
// CHECK-NEXT:   %21 = "arith.muli"(%11, %20) : (index, index) -> index
// CHECK-NEXT:   %22 = "snax.alloc"(%21) <{"memory_space" = 1 : i32, "alignment" = 64 : i32}> : (index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %23 = "builtin.unrealized_conversion_cast"(%22) : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>) -> memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 1 : i32>
// CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  %0 = "test.op"() : () -> (index)
  %1 = "memref.alloc"(%0) <{"alignment" = 64 : i32, "operandSegmentSizes" = array<i32: 1, 0>}> : (index) -> memref<8x?xi32, #tsl.tsl<[2, 4] -> (16, 4), [?, 4] -> (?, 32)>, 1 : i32>
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "test.op"() : () -> index
// CHECK-NEXT:   %1 = "arith.constant"() <{"value" = 8 : index}> : () -> index
// CHECK-NEXT:   %2 = "arith.constant"() <{"value" = 2 : index}> : () -> index
// CHECK-NEXT:   %3 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:   %4 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:   %5 = "arith.divui"(%0, %4) : (index, index) -> index
// CHECK-NEXT:   %6 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:   %7 = "arith.constant"() <{"value" = 32 : index}> : () -> index
// CHECK-NEXT:   %8 = "arith.constant"() <{"value" = 32 : index}> : () -> index
// CHECK-NEXT:   %9 = "arith.muli"(%6, %7) : (index, index) -> index
// CHECK-NEXT:   %10 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:   %11 = "arith.constant"() <{"value" = 16 : index}> : () -> index
// CHECK-NEXT:   %12 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:   %13 = "arith.constant"() <{"value" = 0 : index}> : () -> index
// CHECK-NEXT:   %14 = "arith.muli"(%2, %11) : (index, index) -> index
// CHECK-NEXT:   %15 = "arith.addi"(%13, %14) : (index, index) -> index
// CHECK-NEXT:   %16 = "arith.muli"(%3, %10) : (index, index) -> index
// CHECK-NEXT:   %17 = "arith.addi"(%15, %16) : (index, index) -> index
// CHECK-NEXT:   %18 = "arith.muli"(%5, %9) : (index, index) -> index
// CHECK-NEXT:   %19 = "arith.addi"(%17, %18) : (index, index) -> index
// CHECK-NEXT:   %20 = "arith.muli"(%6, %8) : (index, index) -> index
// CHECK-NEXT:   %21 = "arith.addi"(%19, %20) : (index, index) -> index
// CHECK-NEXT:   %22 = "arith.muli"(%12, %21) : (index, index) -> index
// CHECK-NEXT:   %23 = "snax.alloc"(%22) <{"memory_space" = 1 : i32, "alignment" = 64 : i32}> : (index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %24 = "builtin.unrealized_conversion_cast"(%23) : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>) -> memref<8x?xi32, #tsl.tsl<[2, 4] -> (16, 4), [?, 4] -> (?, 32)>, 1 : i32>
// CHECK-NEXT: }) : () -> ()
