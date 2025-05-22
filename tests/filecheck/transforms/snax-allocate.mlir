// RUN: snax-opt --split-input-file %s -p snax-allocate --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "test.op"() : () -> (index)
  %1 = "snax.alloc"(%0, %0, %0) <{"memory_space" = "L1", "alignment" = 64 : i32}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "test.op"() : () -> index
// CHECK-NEXT:   %1 = "arith.constant"() <{value = 64 : index}> : () -> index
// CHECK-NEXT:   %2 = "func.call"(%0, %1) <{callee = @snax_alloc_l1}> : (index, index) -> !llvm.ptr
// CHECK-NEXT:   %3 = "llvm.load"(%2) : (!llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr)>
// CHECK-NEXT:   %4 = "llvm.extractvalue"(%3) <{position = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr)>) -> !llvm.ptr
// CHECK-NEXT:   %5 = "llvm.extractvalue"(%3) <{position = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr)>) -> !llvm.ptr
// CHECK-NEXT:   %6 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %7 = "llvm.insertvalue"(%6, %4) <{position = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %8 = "llvm.insertvalue"(%7, %5) <{position = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %9 = "arith.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:   %10 = "llvm.insertvalue"(%8, %9) <{position = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %11 = "builtin.unrealized_conversion_cast"(%0) : (index) -> i32
// CHECK-NEXT:   %12 = "llvm.insertvalue"(%10, %11) <{position = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %13 = "builtin.unrealized_conversion_cast"(%0) : (index) -> i32
// CHECK-NEXT:   %14 = "llvm.insertvalue"(%12, %13) <{position = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   "func.func"() <{sym_name = "snax_alloc_l1", function_type = (index, index) -> !llvm.ptr, sym_visibility = "private"}> ({
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()
