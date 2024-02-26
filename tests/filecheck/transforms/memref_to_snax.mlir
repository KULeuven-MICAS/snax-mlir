// RUN: ./compiler/snax-opt --split-input-file %s -p memref-to-snax --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<16x16xi32>
}) : () -> ()

// expect nothing to change because no memory space is specified
// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<16x16xi32>
// CHECK-NEXT: }) : () -> ()


// -----

"builtin.module"() ({
  %0 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<16x16xi32, 1: i32>
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "arith.constant"() <{"value" = 16 : index}> : () -> index
// CHECK-NEXT:   %1 = "arith.constant"() <{"value" = 16 : index}> : () -> index
// CHECK-NEXT:   %2 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:   %3 = "arith.muli"(%0, %2) : (index, index) -> index
// CHECK-NEXT:   %4 = "arith.muli"(%1, %3) : (index, index) -> index
// CHECK-NEXT:   %5 = "snax.alloc"(%4, %0, %1) <{"memory_space" = 1 : i32, "alignment" = 64 : i64}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %6 = "builtin.unrealized_conversion_cast"(%5) : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>) -> memref<16x16xi32, 1 : i32>
// CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  %0 = "test.op"() : () -> (index)
  %1 = "memref.alloc"(%0) <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 1, 0>}> : (index) -> memref<?x16xi32, 1: i32>
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "test.op"() : () -> index
// CHECK-NEXT:   %1 = "arith.constant"() <{"value" = 16 : index}> : () -> index
// CHECK-NEXT:   %2 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:   %3 = "arith.muli"(%0, %2) : (index, index) -> index
// CHECK-NEXT:   %4 = "arith.muli"(%1, %3) : (index, index) -> index
// CHECK-NEXT:   %5 = "snax.alloc"(%4, %0, %1) <{"memory_space" = 1 : i32, "alignment" = 64 : i64}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %6 = "builtin.unrealized_conversion_cast"(%5) : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>) -> memref<?x16xi32, 1 : i32>
// CHECK-NEXT: }) : () -> ()


// -----

"builtin.module"() ({
  %0 = "memref.alloc"() <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 1 : i32>
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
// CHECK-NEXT:   %11 = "arith.constant"() <{"value" = 1 : index}> : () -> index
// CHECK-NEXT:   %12 = "arith.constant"() <{"value" = 0 : index}> : () -> index
// CHECK-NEXT:   %13 = "arith.constant"() <{"value" = 1 : index}> : () -> index
// CHECK-NEXT:   %14 = "arith.subi"(%2, %13) : (index, index) -> index
// CHECK-NEXT:   %15 = "arith.muli"(%14, %10) : (index, index) -> index
// CHECK-NEXT:   %16 = "arith.addi"(%12, %15) : (index, index) -> index
// CHECK-NEXT:   %17 = "arith.subi"(%3, %13) : (index, index) -> index
// CHECK-NEXT:   %18 = "arith.muli"(%17, %9) : (index, index) -> index
// CHECK-NEXT:   %19 = "arith.addi"(%16, %18) : (index, index) -> index
// CHECK-NEXT:   %20 = "arith.subi"(%4, %13) : (index, index) -> index
// CHECK-NEXT:   %21 = "arith.muli"(%20, %8) : (index, index) -> index
// CHECK-NEXT:   %22 = "arith.addi"(%19, %21) : (index, index) -> index
// CHECK-NEXT:   %23 = "arith.subi"(%5, %13) : (index, index) -> index
// CHECK-NEXT:   %24 = "arith.muli"(%23, %7) : (index, index) -> index
// CHECK-NEXT:   %25 = "arith.addi"(%22, %24) : (index, index) -> index
// CHECK-NEXT:   %26 = "arith.addi"(%25, %13) : (index, index) -> index
// CHECK-NEXT:   %27 = "arith.muli"(%11, %26) : (index, index) -> index
// CHECK-NEXT:   %28 = "snax.alloc"(%27, %0, %1) <{"memory_space" = 1 : i32, "alignment" = 64 : i64}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %29 = "builtin.unrealized_conversion_cast"(%28) : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>) -> memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 1 : i32>
// CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  %0 = "test.op"() : () -> (index)
  %1 = "memref.alloc"(%0) <{"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 1, 0>}> : (index) -> memref<8x?xi32, #tsl.tsl<[2, 4] -> (16, 4), [?, 4] -> (?, 32)>, 1 : i32>
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
// CHECK-NEXT:   %12 = "arith.constant"() <{"value" = 1 : index}> : () -> index
// CHECK-NEXT:   %13 = "arith.constant"() <{"value" = 0 : index}> : () -> index
// CHECK-NEXT:   %14 = "arith.constant"() <{"value" = 1 : index}> : () -> index
// CHECK-NEXT:   %15 = "arith.subi"(%2, %14) : (index, index) -> index
// CHECK-NEXT:   %16 = "arith.muli"(%15, %11) : (index, index) -> index
// CHECK-NEXT:   %17 = "arith.addi"(%13, %16) : (index, index) -> index
// CHECK-NEXT:   %18 = "arith.subi"(%3, %14) : (index, index) -> index
// CHECK-NEXT:   %19 = "arith.muli"(%18, %10) : (index, index) -> index
// CHECK-NEXT:   %20 = "arith.addi"(%17, %19) : (index, index) -> index
// CHECK-NEXT:   %21 = "arith.subi"(%5, %14) : (index, index) -> index
// CHECK-NEXT:   %22 = "arith.muli"(%21, %9) : (index, index) -> index
// CHECK-NEXT:   %23 = "arith.addi"(%20, %22) : (index, index) -> index
// CHECK-NEXT:   %24 = "arith.subi"(%6, %14) : (index, index) -> index
// CHECK-NEXT:   %25 = "arith.muli"(%24, %8) : (index, index) -> index
// CHECK-NEXT:   %26 = "arith.addi"(%23, %25) : (index, index) -> index
// CHECK-NEXT:   %27 = "arith.addi"(%26, %14) : (index, index) -> index
// CHECK-NEXT:   %28 = "arith.muli"(%12, %27) : (index, index) -> index
// CHECK-NEXT:   %29 = "snax.alloc"(%28, %1, %0) <{"memory_space" = 1 : i32, "alignment" = 64 : i64}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %30 = "builtin.unrealized_conversion_cast"(%29) : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>) -> memref<8x?xi32, #tsl.tsl<[2, 4] -> (16, 4), [?, 4] -> (?, 32)>, 1 : i32>
// CHECK-NEXT: }) : () -> ()
