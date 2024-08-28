// RUN: ./compiler/snax-opt --split-input-file %s -p double-buffer | filecheck %s

// Nothing should happen with for-loops without copy ins 
builtin.module {
  func.func @streamer_matmul(%arg0 : memref<24x24xi8, "L3">, %arg1 : memref<24x24xi8, strided<[1, 24]>, "L3">, %arg2 : memref<24x24xi32, "L3">) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 0 : index
    %2 = arith.constant 1 : index
    %3 = arith.constant 24 : index
    %4 = arith.constant 8 : index
    %11 = arith.constant 0 : index
    %12 = arith.constant 1 : index
    %13 = memref.alloc(%4, %4) {"alignment" = 64 : i64} : memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">
    scf.for %arg3 = %1 to %3 step %4 {
      scf.for %arg4 = %1 to %3 step %4 {
        %16 = memref.subview %arg2[%arg3, %arg4] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
        "memref.copy"(%13, %16) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[24, 1], offset: ?>>) -> ()
      }
    }
    func.return
  }
}

//CHECK: builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul(%arg0 : memref<24x24xi8, "L3">, %arg1 : memref<24x24xi8, strided<[1, 24]>, "L3">, %arg2 : memref<24x24xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 0 : i32
//CHECK-NEXT:     %1 = arith.constant 0 : index
//CHECK-NEXT:     %2 = arith.constant 1 : index
//CHECK-NEXT:     %3 = arith.constant 24 : index
//CHECK-NEXT:     %4 = arith.constant 8 : index
//CHECK-NEXT:     %5 = arith.constant 0 : index
//CHECK-NEXT:     %6 = arith.constant 1 : index
//CHECK-NEXT:     %7 = memref.alloc(%4, %4) {"alignment" = 64 : i64} : memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">
//CHECK-NEXT:     scf.for %arg3 = %1 to %3 step %4 {
//CHECK-NEXT:       scf.for %arg4 = %1 to %3 step %4 {
//CHECK-NEXT:         %8 = memref.subview %arg2[%arg3, %arg4] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:         "memref.copy"(%7, %8) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[24, 1], offset: ?>>) -> ()
//CHECK-NEXT:       }
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
// -----

//Nothing should happen with for-loops without copy outs
builtin.module {
  func.func @streamer_matmul(%arg0 : memref<24x24xi8, "L3">, %arg1 : memref<24x24xi8, strided<[1, 24]>, "L3">, %arg2 : memref<24x24xi32, "L3">) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 0 : index
    %2 = arith.constant 1 : index
    %3 = arith.constant 24 : index
    %4 = arith.constant 8 : index
    %5 = arith.constant 0 : index
    %6 = arith.constant 1 : index
    %7 = memref.alloc(%4, %3) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
    scf.for %arg3 = %1 to %3 step %4 {
      scf.for %arg4 = %1 to %3 step %4 {
        %14 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<24x24xi8, "L3"> to memref<?x?xi8, strided<[24, 1], offset: ?>>
        "memref.copy"(%14, %7) : (memref<?x?xi8, strided<[24, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
      }
    }
    func.return
  }
}

//CHECK: builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul(%arg0 : memref<24x24xi8, "L3">, %arg1 : memref<24x24xi8, strided<[1, 24]>, "L3">, %arg2 : memref<24x24xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 0 : i32
//CHECK-NEXT:     %1 = arith.constant 0 : index
//CHECK-NEXT:     %2 = arith.constant 1 : index
//CHECK-NEXT:     %3 = arith.constant 24 : index
//CHECK-NEXT:     %4 = arith.constant 8 : index
//CHECK-NEXT:     %5 = arith.constant 0 : index
//CHECK-NEXT:     %6 = arith.constant 1 : index
//CHECK-NEXT:     %7 = memref.alloc(%4, %3) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:     scf.for %arg3 = %1 to %3 step %4 {
//CHECK-NEXT:       scf.for %arg4 = %1 to %3 step %4 {
//CHECK-NEXT:         %8 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<24x24xi8, "L3"> to memref<?x?xi8, strided<[24, 1], offset: ?>>
//CHECK-NEXT:         "memref.copy"(%8, %7) : (memref<?x?xi8, strided<[24, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:       }
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
// -----

//Double buffer should be applied to for-loops with copy ins and outs
builtin.module {
  func.func @streamer_matmul(%arg0 : memref<16x16xi8, "L3">, %arg1 : memref<16x16xi8, strided<[1, 16]>, "L3">, %arg2 : memref<16x16xi32, "L3">) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 0 : index
    %2 = arith.constant 1 : index
    %3 = arith.constant 16 : index
    %4 = arith.constant 8 : index
    %5 = arith.constant 0 : index
    %6 = arith.constant 1 : index
    %7 = memref.alloc(%4, %3) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
    %11 = arith.constant 0 : index
    %12 = arith.constant 1 : index
    %13 = memref.alloc(%4, %4) {"alignment" = 64 : i64} : memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">
    scf.for %arg3 = %1 to %3 step %4 {
      scf.for %arg4 = %1 to %3 step %4 {
        %14 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<16x16xi8, "L3"> to memref<?x?xi8, strided<[16, 1], offset: ?>>
        %16 = memref.subview %arg2[%arg3, %arg4] [%4, %4] [%2, %2] : memref<16x16xi32, "L3"> to memref<?x?xi32, strided<[16, 1], offset: ?>>
        "memref.copy"(%14, %7) : (memref<?x?xi8, strided<[16, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
        "memref.copy"(%13, %16) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[16, 1], offset: ?>>) -> ()
      }
    }
    func.return
  }
}

//CHECK: builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul(%arg0 : memref<16x16xi8, "L3">, %arg1 : memref<16x16xi8, strided<[1, 16]>, "L3">, %arg2 : memref<16x16xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 0 : i32
//CHECK-NEXT:     %1 = arith.constant 0 : index
//CHECK-NEXT:     %2 = arith.constant 1 : index
//CHECK-NEXT:     %3 = arith.constant 16 : index
//CHECK-NEXT:     %4 = arith.constant 8 : index
//CHECK-NEXT:     %5 = arith.constant 0 : index
//CHECK-NEXT:     %6 = arith.constant 1 : index
//CHECK-NEXT:     %7 = memref.alloc(%4, %3) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:     %8 = memref.alloc(%4, %3) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:     %9 = arith.constant 0 : index
//CHECK-NEXT:     %10 = arith.constant 1 : index
//CHECK-NEXT:     %11 = memref.alloc(%4, %4) {"alignment" = 64 : i64} : memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">
//CHECK-NEXT:     %12 = memref.alloc(%4, %4) {"alignment" = 64 : i64} : memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">
//CHECK-NEXT:     scf.for %arg3 = %1 to %3 step %4 {
//CHECK-NEXT:       %13 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<16x16xi8, "L3"> to memref<?x?xi8, strided<[16, 1], offset: ?>>
//CHECK-NEXT:       %14 = memref.subview %arg2[%arg3, %1] [%4, %4] [%2, %2] : memref<16x16xi32, "L3"> to memref<?x?xi32, strided<[16, 1], offset: ?>>
//CHECK-NEXT:       %15 = memref.subview %arg2[%arg3, %1] [%4, %4] [%2, %2] : memref<16x16xi32, "L3"> to memref<?x?xi32, strided<[16, 1], offset: ?>>
//CHECK-NEXT:       "memref.copy"(%13, %7) : (memref<?x?xi8, strided<[16, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       %16 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<16x16xi8, "L3"> to memref<?x?xi8, strided<[16, 1], offset: ?>>
//CHECK-NEXT:       %17 = memref.subview %arg2[%arg3, %4] [%4, %4] [%2, %2] : memref<16x16xi32, "L3"> to memref<?x?xi32, strided<[16, 1], offset: ?>>
//CHECK-NEXT:       %18 = memref.subview %arg2[%arg3, %4] [%4, %4] [%2, %2] : memref<16x16xi32, "L3"> to memref<?x?xi32, strided<[16, 1], offset: ?>>
//CHECK-NEXT:       "memref.copy"(%16, %8) : (memref<?x?xi8, strided<[16, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       %19 = arith.subi %3, %4 : index
//CHECK-NEXT:       %20 = arith.constant 2 : index
//CHECK-NEXT:       %21 = arith.muli %4, %20 : index
//CHECK-NEXT:       %22 = arith.subi %3, %21 : index
//CHECK-NEXT:       %23 = arith.constant 3 : index
//CHECK-NEXT:       %24 = arith.muli %4, %23 : index
//CHECK-NEXT:       %25 = arith.subi %3, %24 : index
//CHECK-NEXT:       scf.for %arg4 = %21 to %3 step %21 {
//CHECK-NEXT:         %26 = arith.subi %arg4, %21 : index
//CHECK-NEXT:         %27 = arith.subi %arg4, %4 : index
//CHECK-NEXT:         %28 = arith.addi %arg4, %4 : index
//CHECK-NEXT:         %29 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<16x16xi8, "L3"> to memref<?x?xi8, strided<[16, 1], offset: ?>>
//CHECK-NEXT:         %30 = memref.subview %arg2[%arg3, %arg4] [%4, %4] [%2, %2] : memref<16x16xi32, "L3"> to memref<?x?xi32, strided<[16, 1], offset: ?>>
//CHECK-NEXT:         %31 = memref.subview %arg2[%arg3, %26] [%4, %4] [%2, %2] : memref<16x16xi32, "L3"> to memref<?x?xi32, strided<[16, 1], offset: ?>>
//CHECK-NEXT:         "memref.copy"(%11, %31) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[16, 1], offset: ?>>) -> ()
//CHECK-NEXT:         "memref.copy"(%29, %7) : (memref<?x?xi8, strided<[16, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:         "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:         %32 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<16x16xi8, "L3"> to memref<?x?xi8, strided<[16, 1], offset: ?>>
//CHECK-NEXT:         %33 = memref.subview %arg2[%arg3, %arg4] [%4, %4] [%2, %2] : memref<16x16xi32, "L3"> to memref<?x?xi32, strided<[16, 1], offset: ?>>
//CHECK-NEXT:         %34 = memref.subview %arg2[%arg3, %27] [%4, %4] [%2, %2] : memref<16x16xi32, "L3"> to memref<?x?xi32, strided<[16, 1], offset: ?>>
//CHECK-NEXT:         "memref.copy"(%12, %34) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[16, 1], offset: ?>>) -> ()
//CHECK-NEXT:         "memref.copy"(%32, %8) : (memref<?x?xi8, strided<[16, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:         "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       }
//CHECK-NEXT:       %35 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<16x16xi8, "L3"> to memref<?x?xi8, strided<[16, 1], offset: ?>>
//CHECK-NEXT:       %36 = memref.subview %arg2[%arg3, %22] [%4, %4] [%2, %2] : memref<16x16xi32, "L3"> to memref<?x?xi32, strided<[16, 1], offset: ?>>
//CHECK-NEXT:       "memref.copy"(%11, %36) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[16, 1], offset: ?>>) -> ()
//CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       %37 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<16x16xi8, "L3"> to memref<?x?xi8, strided<[16, 1], offset: ?>>
//CHECK-NEXT:       %38 = memref.subview %arg2[%arg3, %19] [%4, %4] [%2, %2] : memref<16x16xi32, "L3"> to memref<?x?xi32, strided<[16, 1], offset: ?>>
//CHECK-NEXT:       "memref.copy"(%12, %38) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[16, 1], offset: ?>>) -> ()
//CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
// -----

// Double buffer should be applied to for-loops with copy ins and outs when an uneven iteration amount is used
builtin.module {
  func.func @streamer_matmul(%arg0 : memref<24x24xi8, "L3">, %arg1 : memref<24x24xi8, strided<[1, 24]>, "L3">, %arg2 : memref<24x24xi32, "L3">) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 0 : index
    %2 = arith.constant 1 : index
    %3 = arith.constant 24 : index
    %4 = arith.constant 8 : index
    %5 = arith.constant 0 : index
    %6 = arith.constant 1 : index
    %7 = memref.alloc(%4, %3) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
    %11 = arith.constant 0 : index
    %12 = arith.constant 1 : index
    %13 = memref.alloc(%4, %4) {"alignment" = 64 : i64} : memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">
    scf.for %arg3 = %1 to %3 step %4 {
      scf.for %arg4 = %1 to %3 step %4 {
        %14 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<24x24xi8, "L3"> to memref<?x?xi8, strided<[24, 1], offset: ?>>
        %16 = memref.subview %arg2[%arg3, %arg4] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
        "memref.copy"(%14, %7) : (memref<?x?xi8, strided<[24, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
        "memref.copy"(%13, %16) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[24, 1], offset: ?>>) -> ()
      }
    }
    func.return
  }
}

//CHECK: builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul(%arg0 : memref<24x24xi8, "L3">, %arg1 : memref<24x24xi8, strided<[1, 24]>, "L3">, %arg2 : memref<24x24xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 0 : i32
//CHECK-NEXT:     %1 = arith.constant 0 : index
//CHECK-NEXT:     %2 = arith.constant 1 : index
//CHECK-NEXT:     %3 = arith.constant 24 : index
//CHECK-NEXT:     %4 = arith.constant 8 : index
//CHECK-NEXT:     %5 = arith.constant 0 : index
//CHECK-NEXT:     %6 = arith.constant 1 : index
//CHECK-NEXT:     %7 = memref.alloc(%4, %3) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:     %8 = memref.alloc(%4, %3) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:     %9 = arith.constant 0 : index
//CHECK-NEXT:     %10 = arith.constant 1 : index
//CHECK-NEXT:     %11 = memref.alloc(%4, %4) {"alignment" = 64 : i64} : memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">
//CHECK-NEXT:     %12 = memref.alloc(%4, %4) {"alignment" = 64 : i64} : memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">
//CHECK-NEXT:     scf.for %arg3 = %1 to %3 step %4 {
//CHECK-NEXT:       %13 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<24x24xi8, "L3"> to memref<?x?xi8, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       %14 = memref.subview %arg2[%arg3, %1] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       %15 = memref.subview %arg2[%arg3, %1] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       "memref.copy"(%13, %7) : (memref<?x?xi8, strided<[24, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       %16 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<24x24xi8, "L3"> to memref<?x?xi8, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       %17 = memref.subview %arg2[%arg3, %4] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       %18 = memref.subview %arg2[%arg3, %4] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       "memref.copy"(%16, %8) : (memref<?x?xi8, strided<[24, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       %19 = arith.subi %3, %4 : index
//CHECK-NEXT:       %20 = arith.constant 2 : index
//CHECK-NEXT:       %21 = arith.muli %4, %20 : index
//CHECK-NEXT:       %22 = arith.subi %3, %21 : index
//CHECK-NEXT:       %23 = arith.constant 3 : index
//CHECK-NEXT:       %24 = arith.muli %4, %23 : index
//CHECK-NEXT:       %25 = arith.subi %3, %24 : index
//CHECK-NEXT:       scf.for %arg4 = %21 to %22 step %21 {
//CHECK-NEXT:         %26 = arith.subi %arg4, %21 : index
//CHECK-NEXT:         %27 = arith.subi %arg4, %4 : index
//CHECK-NEXT:         %28 = arith.addi %arg4, %4 : index
//CHECK-NEXT:         %29 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<24x24xi8, "L3"> to memref<?x?xi8, strided<[24, 1], offset: ?>>
//CHECK-NEXT:         %30 = memref.subview %arg2[%arg3, %arg4] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:         %31 = memref.subview %arg2[%arg3, %26] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:         "memref.copy"(%11, %31) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[24, 1], offset: ?>>) -> ()
//CHECK-NEXT:         "memref.copy"(%29, %7) : (memref<?x?xi8, strided<[24, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:         "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:         %32 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<24x24xi8, "L3"> to memref<?x?xi8, strided<[24, 1], offset: ?>>
//CHECK-NEXT:         %33 = memref.subview %arg2[%arg3, %arg4] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:         %34 = memref.subview %arg2[%arg3, %27] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:         "memref.copy"(%12, %34) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[24, 1], offset: ?>>) -> ()
//CHECK-NEXT:         "memref.copy"(%32, %8) : (memref<?x?xi8, strided<[24, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:         "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       }
//CHECK-NEXT:       %35 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<24x24xi8, "L3"> to memref<?x?xi8, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       %36 = memref.subview %arg2[%arg3, %25] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       "memref.copy"(%11, %36) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[24, 1], offset: ?>>) -> ()
//CHECK-NEXT:       "memref.copy"(%35, %7) : (memref<?x?xi8, strided<[24, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       %37 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<24x24xi8, "L3"> to memref<?x?xi8, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       %38 = memref.subview %arg2[%arg3, %22] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       "memref.copy"(%12, %38) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[24, 1], offset: ?>>) -> ()
//CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       %39 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<24x24xi8, "L3"> to memref<?x?xi8, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       %40 = memref.subview %arg2[%arg3, %19] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       "memref.copy"(%11, %40) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[24, 1], offset: ?>>) -> ()
//CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }

