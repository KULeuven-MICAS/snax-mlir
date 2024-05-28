// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_SINGLETRIP

"builtin.module"() ({
  %0 = "test.op"() : () -> memref<8x8xi32, strided<[1, 8]>, 1 : i32>
  %1 = "snax.layout_cast"(%0) : (memref<8x8xi32, strided<[1, 8]>, 1 : i32>) -> memref<8x8xi32, strided<[1, 16]>, 1 : i32>
  %2 = "test.op"() : () -> index
  %3 = "snax.alloc"(%2, %2, %2) <{"memory_space" = 0 : i32, "alignment" = 64 : i32}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
}) : () -> ()


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> memref<8x8xi32, strided<[1, 8]>, 1 : i32>
// CHECK-NEXT:   %1 = "snax.layout_cast"(%0) : (memref<8x8xi32, strided<[1, 8]>, 1 : i32>) -> memref<8x8xi32, strided<[1, 16]>, 1 : i32>
// CHECK-NEXT:   %2 = "test.op"() : () -> index
// CHECK-NEXT:   %3 = "snax.alloc"(%2, %2, %2) <{"memory_space" = 0 : i32, "alignment" = 64 : i32}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT: }
