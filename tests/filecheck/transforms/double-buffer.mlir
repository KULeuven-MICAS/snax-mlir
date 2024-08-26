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
//CHECK-NEXT:       "memref.copy"(%13, %7) : (memref<?x?xi8, strided<[16, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       %15 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<16x16xi8, "L3"> to memref<?x?xi8, strided<[16, 1], offset: ?>>
//CHECK-NEXT:       %16 = memref.subview %arg2[%arg3, %4] [%4, %4] [%2, %2] : memref<16x16xi32, "L3"> to memref<?x?xi32, strided<[16, 1], offset: ?>>
//CHECK-NEXT:       "memref.copy"(%15, %8) : (memref<?x?xi8, strided<[16, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       %17 = arith.subi %3, %4 : index
//CHECK-NEXT:       %18 = arith.constant 2 : index
//CHECK-NEXT:       %19 = arith.muli %4, %18 : index
//CHECK-NEXT:       %20 = arith.subi %3, %19 : index
//CHECK-NEXT:       %21 = arith.constant 3 : index
//CHECK-NEXT:       %22 = arith.muli %4, %21 : index
//CHECK-NEXT:       %23 = arith.subi %3, %22 : index
//CHECK-NEXT:       scf.for %arg4 = %19 to %3 step %19 {
//CHECK-NEXT:         %24 = arith.subi %arg4, %19 : index
//CHECK-NEXT:         %25 = arith.subi %arg4, %4 : index
//CHECK-NEXT:         %26 = arith.addi %arg4, %4 : index
//CHECK-NEXT:         %27 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<16x16xi8, "L3"> to memref<?x?xi8, strided<[16, 1], offset: ?>>
//CHECK-NEXT:         %28 = memref.subview %arg2[%arg3, %24] [%4, %4] [%2, %2] : memref<16x16xi32, "L3"> to memref<?x?xi32, strided<[16, 1], offset: ?>>
//CHECK-NEXT:         "memref.copy"(%27, %7) : (memref<?x?xi8, strided<[16, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:         "memref.copy"(%11, %28) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[16, 1], offset: ?>>) -> ()
//CHECK-NEXT:         "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:         %29 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<16x16xi8, "L3"> to memref<?x?xi8, strided<[16, 1], offset: ?>>
//CHECK-NEXT:         %30 = memref.subview %arg2[%arg3, %25] [%4, %4] [%2, %2] : memref<16x16xi32, "L3"> to memref<?x?xi32, strided<[16, 1], offset: ?>>
//CHECK-NEXT:         "memref.copy"(%29, %8) : (memref<?x?xi8, strided<[16, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:         "memref.copy"(%12, %30) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[16, 1], offset: ?>>) -> ()
//CHECK-NEXT:         "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       }
//CHECK-NEXT:       %31 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<16x16xi8, "L3"> to memref<?x?xi8, strided<[16, 1], offset: ?>>
//CHECK-NEXT:       %32 = memref.subview %arg2[%arg3, %20] [%4, %4] [%2, %2] : memref<16x16xi32, "L3"> to memref<?x?xi32, strided<[16, 1], offset: ?>>
//CHECK-NEXT:       "memref.copy"(%12, %32) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[16, 1], offset: ?>>) -> ()
//CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       %33 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<16x16xi8, "L3"> to memref<?x?xi8, strided<[16, 1], offset: ?>>
//CHECK-NEXT:       %34 = memref.subview %arg2[%arg3, %17] [%4, %4] [%2, %2] : memref<16x16xi32, "L3"> to memref<?x?xi32, strided<[16, 1], offset: ?>>
//CHECK-NEXT:       "memref.copy"(%11, %34) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[16, 1], offset: ?>>) -> ()
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
//CHECK-NEXT:       "memref.copy"(%13, %7) : (memref<?x?xi8, strided<[24, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       %15 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<24x24xi8, "L3"> to memref<?x?xi8, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       %16 = memref.subview %arg2[%arg3, %4] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       "memref.copy"(%15, %8) : (memref<?x?xi8, strided<[24, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       %17 = arith.subi %3, %4 : index
//CHECK-NEXT:       %18 = arith.constant 2 : index
//CHECK-NEXT:       %19 = arith.muli %4, %18 : index
//CHECK-NEXT:       %20 = arith.subi %3, %19 : index
//CHECK-NEXT:       %21 = arith.constant 3 : index
//CHECK-NEXT:       %22 = arith.muli %4, %21 : index
//CHECK-NEXT:       %23 = arith.subi %3, %22 : index
//CHECK-NEXT:       scf.for %arg4 = %19 to %3 step %19 {
//CHECK-NEXT:         %24 = arith.subi %arg4, %19 : index
//CHECK-NEXT:         %25 = arith.subi %arg4, %4 : index
//CHECK-NEXT:         %26 = arith.addi %arg4, %4 : index
//CHECK-NEXT:         %27 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<24x24xi8, "L3"> to memref<?x?xi8, strided<[24, 1], offset: ?>>
//CHECK-NEXT:         %28 = memref.subview %arg2[%arg3, %24] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:         "memref.copy"(%27, %7) : (memref<?x?xi8, strided<[24, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:         "memref.copy"(%11, %28) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[24, 1], offset: ?>>) -> ()
//CHECK-NEXT:         "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:         %29 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<24x24xi8, "L3"> to memref<?x?xi8, strided<[24, 1], offset: ?>>
//CHECK-NEXT:         %30 = memref.subview %arg2[%arg3, %25] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:         "memref.copy"(%29, %8) : (memref<?x?xi8, strided<[24, 1], offset: ?>>, memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">) -> ()
//CHECK-NEXT:         "memref.copy"(%12, %30) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[24, 1], offset: ?>>) -> ()
//CHECK-NEXT:         "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       }
//CHECK-NEXT:       %31 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<24x24xi8, "L3"> to memref<?x?xi8, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       %32 = memref.subview %arg2[%arg3, %20] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       "memref.copy"(%12, %32) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[24, 1], offset: ?>>) -> ()
//CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:       %33 = memref.subview %arg0[%arg3, %1] [%4, %3] [%2, %2] : memref<24x24xi8, "L3"> to memref<?x?xi8, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       %34 = memref.subview %arg2[%arg3, %17] [%4, %4] [%2, %2] : memref<24x24xi32, "L3"> to memref<?x?xi32, strided<[24, 1], offset: ?>>
//CHECK-NEXT:       "memref.copy"(%11, %34) : (memref<?x?xi32, #tsl.tsl<[?, 8] -> (?, 32), [?, 8] -> (256, 4)>, "L1">, memref<?x?xi32, strided<[24, 1], offset: ?>>) -> ()
//CHECK-NEXT:       "snax.cluster_sync_op"() : () -> ()
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }

