// RUN: snax-opt --split-input-file %s -p snax-allocate{mode=minimalloc} | filecheck %s

builtin.module {
  func.func public @test() {
    %0 = arith.constant 5 : index
    %1 = arith.constant 13 : index
    %2 = "snax.alloc"(%1, %0, %0) <{memory_space = "Test", alignment = 10 : i32}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %3 = "builtin.unrealized_conversion_cast" (%2) : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>) ->  memref<5x5xi32>
    %4 = "snax.alloc"(%1, %0, %0) <{memory_space = "Test", alignment = 10 : i32}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %5 = "builtin.unrealized_conversion_cast" (%4) : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>) -> memref<5x5xi32>
    "test.op"(%3) : (memref<5x5xi32>) -> ()
    "test.op"(%5) : (memref<5x5xi32>) -> ()
    %6 = "snax.alloc"(%1, %0, %0) <{memory_space = "Test", alignment = 14 : i32}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %7 = "builtin.unrealized_conversion_cast" (%6) : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>) -> memref<5x5xi32>
    "test.op"(%7) : (memref<5x5xi32>) -> ()
    func.return
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func public @test() {
// CHECK-NEXT:     %0 = arith.constant 5 : index
// CHECK-NEXT:     %1 = arith.constant 13 : index
//                    First allocation at 0:
// CHECK-NEXT:     %2 = arith.constant 0 : i32
// CHECK-NEXT:     %3 = "llvm.inttoptr"(%2) : (i32) -> !llvm.ptr
// CHECK-NEXT:     %4 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %5 = "llvm.insertvalue"(%4, %3) <{position = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %6 = "llvm.insertvalue"(%5, %3) <{position = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %7 = arith.constant 0 : i32
// CHECK-NEXT:     %8 = "llvm.insertvalue"(%6, %7) <{position = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %9 = builtin.unrealized_conversion_cast %0 : index to i32
// CHECK-NEXT:     %10 = "llvm.insertvalue"(%8, %9) <{position = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %11 = builtin.unrealized_conversion_cast %0 : index to i32
// CHECK-NEXT:     %12 = "llvm.insertvalue"(%10, %11) <{position = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %13 = builtin.unrealized_conversion_cast %12 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)> to memref<5x5xi32>
//
//                    Second allocation at 20:
// CHECK-NEXT:     %14 = arith.constant 20 : i32
// CHECK-NEXT:     %15 = "llvm.inttoptr"(%14) : (i32) -> !llvm.ptr
// CHECK-NEXT:     %16 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %17 = "llvm.insertvalue"(%16, %15) <{position = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %18 = "llvm.insertvalue"(%17, %15) <{position = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %19 = arith.constant 0 : i32
// CHECK-NEXT:     %20 = "llvm.insertvalue"(%18, %19) <{position = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %21 = builtin.unrealized_conversion_cast %0 : index to i32
// CHECK-NEXT:     %22 = "llvm.insertvalue"(%20, %21) <{position = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %23 = builtin.unrealized_conversion_cast %0 : index to i32
// CHECK-NEXT:     %24 = "llvm.insertvalue"(%22, %23) <{position = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %25 = builtin.unrealized_conversion_cast %24 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)> to memref<5x5xi32>
// CHECK-NEXT:     "test.op"(%13) : (memref<5x5xi32>) -> ()
// CHECK-NEXT:     memref.dealloc %13 : memref<5x5xi32>
// CHECK-NEXT:     "test.op"(%25) : (memref<5x5xi32>) -> ()
// CHECK-NEXT:     memref.dealloc %25 : memref<5x5xi32>
//
//                    Third allocation back at 0:
// CHECK-NEXT:     %26 = arith.constant 0 : i32
// CHECK-NEXT:     %27 = "llvm.inttoptr"(%26) : (i32) -> !llvm.ptr
// CHECK-NEXT:     %28 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %29 = "llvm.insertvalue"(%28, %27) <{position = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %30 = "llvm.insertvalue"(%29, %27) <{position = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %31 = arith.constant 0 : i32
// CHECK-NEXT:     %32 = "llvm.insertvalue"(%30, %31) <{position = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %33 = builtin.unrealized_conversion_cast %0 : index to i32
// CHECK-NEXT:     %34 = "llvm.insertvalue"(%32, %33) <{position = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %35 = builtin.unrealized_conversion_cast %0 : index to i32
// CHECK-NEXT:     %36 = "llvm.insertvalue"(%34, %35) <{position = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:     %37 = builtin.unrealized_conversion_cast %36 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)> to memref<5x5xi32>
// CHECK-NEXT:     "test.op"(%37) : (memref<5x5xi32>) -> ()
// CHECK-NEXT:     memref.dealloc %37 : memref<5x5xi32>
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
