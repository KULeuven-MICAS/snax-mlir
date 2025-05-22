// RUN: snax-opt --split-input-file %s -p snax-allocate{mode=static} | filecheck %s

"builtin.module"() ({
  %0 = arith.constant 5 : index
  %1 = arith.constant 13 : index
  %2 = "snax.alloc"(%1, %0, %0) <{"memory_space" = "Test", "alignment" = 10 : i32}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
  %3 = "snax.alloc"(%1, %0, %0) <{"memory_space" = "Test", "alignment" = 10 : i32}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
  %4 = "snax.alloc"(%1, %0, %0) <{"memory_space" = "Test", "alignment" = 14 : i32}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = arith.constant 5 : index
// CHECK-NEXT:   %1 = arith.constant 13 : index

//                    First allocation at 0:
// CHECK-NEXT:   %2 = arith.constant 0 : i32
// CHECK-NEXT:   %3 = "llvm.inttoptr"(%2) : (i32) -> !llvm.ptr<!llvm.ptr>
// CHECK-NEXT:   %4 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %5 = "llvm.insertvalue"(%4, %3) <{position = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr<!llvm.ptr>) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %6 = "llvm.insertvalue"(%5, %3) <{position = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr<!llvm.ptr>) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %7 = arith.constant 0 : i32
// CHECK-NEXT:   %8 = "llvm.insertvalue"(%6, %7) <{position = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %9 = builtin.unrealized_conversion_cast %0 : index to i32
// CHECK-NEXT:   %10 = "llvm.insertvalue"(%8, %9) <{position = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %11 = builtin.unrealized_conversion_cast %0 : index to i32
// CHECK-NEXT:   %12 = "llvm.insertvalue"(%10, %11) <{position = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
//
//                     Second allocation at 20:
// CHECK-NEXT:   %13 = arith.constant 20 : i32
// CHECK-NEXT:   %14 = "llvm.inttoptr"(%13) : (i32) -> !llvm.ptr<!llvm.ptr>
// CHECK-NEXT:   %15 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %16 = "llvm.insertvalue"(%15, %14) <{position = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr<!llvm.ptr>) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %17 = "llvm.insertvalue"(%16, %14) <{position = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr<!llvm.ptr>) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %18 = arith.constant 0 : i32
// CHECK-NEXT:   %19 = "llvm.insertvalue"(%17, %18) <{position = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %20 = builtin.unrealized_conversion_cast %0 : index to i32
// CHECK-NEXT:   %21 = "llvm.insertvalue"(%19, %20) <{position = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %22 = builtin.unrealized_conversion_cast %0 : index to i32
// CHECK-NEXT:   %23 = "llvm.insertvalue"(%21, %22) <{position = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
//
//                     Third allocation at 20:
// CHECK-NEXT:   %24 = arith.constant 42 : i32
// CHECK-NEXT:   %25 = "llvm.inttoptr"(%24) : (i32) -> !llvm.ptr<!llvm.ptr>
// CHECK-NEXT:   %26 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %27 = "llvm.insertvalue"(%26, %25) <{position = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr<!llvm.ptr>) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %28 = "llvm.insertvalue"(%27, %25) <{position = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr<!llvm.ptr>) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %29 = arith.constant 0 : i32
// CHECK-NEXT:   %30 = "llvm.insertvalue"(%28, %29) <{position = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %31 = builtin.unrealized_conversion_cast %0 : index to i32
// CHECK-NEXT:   %32 = "llvm.insertvalue"(%30, %31) <{position = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %33 = builtin.unrealized_conversion_cast %0 : index to i32
// CHECK-NEXT:   %34 = "llvm.insertvalue"(%32, %33) <{position = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT: }
