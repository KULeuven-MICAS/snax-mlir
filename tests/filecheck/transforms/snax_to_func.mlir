// RUN: ./compiler/snax-opt --split-input-file %s -p snax-to-func --print-op-generic | filecheck %s

"builtin.module"() ({
  "snax.cluster_sync_op"() : () -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   "func.call"() <{"callee" = @snax_cluster_hw_barrier}> : () -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snax_cluster_hw_barrier", "function_type" = () -> (), "sym_visibility" = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  %0 = "test.op"() : () -> (index)
  %1 = "snax.alloc"(%0, %0, %0) <{"memory_space" = 1 : i32, "alignment" = 64 : i32}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "test.op"() : () -> index
// CHECK-NEXT:   %1 = "func.call"(%0) <{"callee" = @snax_alloc_l1}> : (index) -> !llvm.ptr
// CHECK-NEXT:   %2 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %3 = "llvm.insertvalue"(%2, %1) <{"position" = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %4 = "llvm.insertvalue"(%3, %1) <{"position" = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %5 = "arith.constant"() <{"value" = 0 : i32}> : () -> i32
// CHECK-NEXT:   %6 = "llvm.insertvalue"(%4, %5) <{"position" = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %7 = "builtin.unrealized_conversion_cast"(%0) : (index) -> i32
// CHECK-NEXT:   %8 = "llvm.insertvalue"(%6, %7) <{"position" = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %9 = "builtin.unrealized_conversion_cast"(%0) : (index) -> i32
// CHECK-NEXT:   %10 = "llvm.insertvalue"(%8, %9) <{"position" = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   "func.func"() <{"sym_name" = "snax_alloc_l1", "function_type" = (index) -> !llvm.ptr, "sym_visibility" = "private"}> ({
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()
