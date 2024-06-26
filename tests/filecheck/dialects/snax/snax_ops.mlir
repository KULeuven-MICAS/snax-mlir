// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_SINGLETRIP

"builtin.module"() ({
  %0 = "test.op"() : () -> memref<8x8xi32, strided<[1, 8]>, "L1">
  %1 = "snax.layout_cast"(%0) : (memref<8x8xi32, strided<[1, 8]>, "L1">) -> memref<8x8xi32, strided<[1, 16]>, "L1">
  %2 = "snax.mcycle"() : () -> i32
  %3 = "test.op"() : () -> index
  %4 = "snax.alloc"(%3, %3, %3) <{"memory_space" = "L3", "alignment" = 64 : i32}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
}) : () -> ()


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> memref<8x8xi32, strided<[1, 8]>, "L1">
// CHECK-NEXT:   %1 = "snax.layout_cast"(%0) : (memref<8x8xi32, strided<[1, 8]>, "L1">) -> memref<8x8xi32, strided<[1, 16]>, "L1">
// CHECK-NEXT:   %2 = "snax.mcycle"() : () -> i32
// CHECK-NEXT:   %3 = "test.op"() : () -> index
// CHECK-NEXT:   %4 = "snax.alloc"(%3, %3, %3) <{"memory_space" = "L3", "alignment" = 64 : i32}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT: }
