// RUN: ./compiler/snax-opt --split-input-file %s -p reuse-memref-space | filecheck %s

builtin.module {
  func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
    %0 = arith.constant 8 : index
    %1 = arith.constant 1 : index
    %2 = arith.constant 0 : i32
    %3 = arith.constant 0 : index
    scf.for %arg2 = %3 to %0 step %1{
        %4 = memref.alloc(%0, %0) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
        }
    func.return
  }
}
// Check if the memref.alloc is moved out of the loop with all operands outside loop
//CHECK: builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 8 : index
//CHECK-NEXT:     %1 = arith.constant 1 : index
//CHECK-NEXT:     %2 = arith.constant 0 : i32
//CHECK-NEXT:     %3 = arith.constant 0 : index
//CHECK-NEXT:     %4 = memref.alloc(%0, %0) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:     scf.for %arg2 = %3 to %0 step %1 {
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
// -----


builtin.module {
  func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
    %0 = arith.constant 8 : index
    %1 = arith.constant 1 : index
    %2 = arith.constant 0 : i32
    %3 = arith.constant 0 : index
    scf.for %arg2 = %3 to %0 step %1{
        %4 = arith.constant 16 : index
        %5 = arith.constant 8 : index
        %6 = memref.alloc(%4, %5) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
        }
    func.return
  }
}

// Check if the memref.alloc is moved out of the loop with all operands inside loop
//CHECK: builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 8 : index
//CHECK-NEXT:     %1 = arith.constant 1 : index
//CHECK-NEXT:     %2 = arith.constant 0 : i32
//CHECK-NEXT:     %3 = arith.constant 0 : index
//CHECK-NEXT:     %4 = arith.constant 16 : index
//CHECK-NEXT:     %5 = arith.constant 8 : index
//CHECK-NEXT:     %6 = memref.alloc(%4, %5) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:     scf.for %arg2 = %3 to %0 step %1 {
//CHECK-NEXT:       %7 = arith.constant 16 : index
//CHECK-NEXT:       %8 = arith.constant 8 : index
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
// -----

builtin.module {
  func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
    %0 = arith.constant 8 : index
    %1 = arith.constant 1 : index
    %2 = arith.constant 0 : i32
    %3 = arith.constant 0 : index
    scf.for %arg2 = %3 to %0 step %1{
        %5 = arith.constant 8 : index
        %6 = memref.alloc(%arg2, %5) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
        }
    func.return
  }
}

// Nothing should happen to the memref.alloc when sizes are not constant throughout iterations
//CHECK: builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 8 : index
//CHECK-NEXT:     %1 = arith.constant 1 : index
//CHECK-NEXT:     %2 = arith.constant 0 : i32
//CHECK-NEXT:     %3 = arith.constant 0 : index
//CHECK-NEXT:     scf.for %arg2 = %3 to %0 step %1 {
//CHECK-NEXT:       %4 = arith.constant 8 : index
//CHECK-NEXT:       %5 = memref.alloc(%arg2, %4) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
// -----

builtin.module {
  func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
    %0 = arith.constant 8 : index
    %1 = arith.constant 1 : index
    %2 = arith.constant 0 : i32
    %3 = arith.constant 0 : index
    %4 = "memref.dim"(%arg0, %3) :(memref<?x?xi8, "L3">, index) -> index
    %5 = "memref.dim"(%arg0, %1) :(memref<?x?xi8, "L3">, index) -> index
    scf.for %arg2 = %3 to %0 step %1{
        %6 = memref.alloc(%4, %5) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
        }
    func.return
  }
}
// Check if the memref.alloc is moved out of the loop when dim operations are outside loop
//CHECK: builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 8 : index
//CHECK-NEXT:     %1 = arith.constant 1 : index
//CHECK-NEXT:     %2 = arith.constant 0 : i32
//CHECK-NEXT:     %3 = arith.constant 0 : index
//CHECK-NEXT:     %4 = "memref.dim"(%arg0, %3) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:     %5 = "memref.dim"(%arg0, %1) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:     %6 = memref.alloc(%4, %5) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:     scf.for %arg2 = %3 to %0 step %1 {
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
// -----

builtin.module {
  func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
    %0 = arith.constant 8 : index
    %1 = arith.constant 1 : index
    %2 = arith.constant 0 : i32
    %3 = arith.constant 0 : index
    scf.for %arg2 = %3 to %0 step %1{
        %4 = "memref.dim"(%arg0, %3) :(memref<?x?xi8, "L3">, index) -> index
        %5 = "memref.dim"(%arg0, %1) :(memref<?x?xi8, "L3">, index) -> index
        %6 = memref.alloc(%4, %5) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
        }
    func.return
  }
}

// Check if the memref.alloc is elevated outside together with dim operations inside loop
//CHECK: builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 8 : index
//CHECK-NEXT:     %1 = arith.constant 1 : index
//CHECK-NEXT:     %2 = arith.constant 0 : i32
//CHECK-NEXT:     %3 = arith.constant 0 : index
//CHECK-NEXT:     %4 = arith.constant 0 : index
//CHECK-NEXT:     %5 = "memref.dim"(%arg0, %4) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:     %6 = arith.constant 1 : index
//CHECK-NEXT:     %7 = "memref.dim"(%arg0, %6) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:     %8 = memref.alloc(%5, %7) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:     scf.for %arg2 = %3 to %0 step %1 {
//CHECK-NEXT:       %9 = "memref.dim"(%arg0, %3) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:       %10 = "memref.dim"(%arg0, %1) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
// -----

builtin.module {
  func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
    %0 = arith.constant 8 : index
    %1 = arith.constant 1 : index
    %2 = arith.constant 0 : i32
    %3 = arith.constant 0 : index
    scf.for %arg2 = %3 to %0 step %1{
        %4 = "affine.min"(%1, %0) <{"map" = affine_map<(d0)[s0] -> (8, ((d0 * -1) + s0))>}> : (index, index) -> index
        %5 = "memref.dim"(%arg0, %1) :(memref<?x?xi8, "L3">, index) -> index
        %6 = memref.alloc(%4, %5) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
        }
    func.return
  }
}

// Check if the memref.alloc is moved out of the loop when affine.min is inside loop
// so that a new arith.constant is created equal to maximum of affine.min
//CHECK: builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 8 : index
//CHECK-NEXT:     %1 = arith.constant 1 : index
//CHECK-NEXT:     %2 = arith.constant 0 : i32
//CHECK-NEXT:     %3 = arith.constant 0 : index
//CHECK-NEXT:     %4 = arith.constant 8 : index
//CHECK-NEXT:     %5 = arith.constant 1 : index
//CHECK-NEXT:     %6 = "memref.dim"(%arg0, %5) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:     %7 = memref.alloc(%4, %6) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:     scf.for %arg2 = %3 to %0 step %1 {
//CHECK-NEXT:       %8 = "affine.min"(%1, %0) <{"map" = affine_map<(d0)[s0] -> (8, ((d0 * -1) + s0))>}> : (index, index) -> index
//CHECK-NEXT:       %9 = "memref.dim"(%arg0, %1) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
// -----

builtin.module {
  func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
    %0 = arith.constant 8 : index
    %1 = arith.constant 1 : index
    %2 = arith.constant 0 : i32
    %3 = arith.constant 0 : index
    scf.for %arg2 = %3 to %0 step %1{
        scf.for %arg3 = %3 to %0 step %1{
            %4 = "affine.min"(%1, %0) <{"map" = affine_map<(d0)[s0] -> (8, ((d0 * -1) + s0))>}> : (index, index) -> index
            %5 = "memref.dim"(%arg0, %1) :(memref<?x?xi8, "L3">, index) -> index
            %6 = memref.alloc(%4, %5) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
            }
        }
    func.return
  }
}

//CHECK: builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 8 : index
//CHECK-NEXT:     %1 = arith.constant 1 : index
//CHECK-NEXT:     %2 = arith.constant 0 : i32
//CHECK-NEXT:     %3 = arith.constant 0 : index
//CHECK-NEXT:     %4 = arith.constant 8 : index
//CHECK-NEXT:     %5 = arith.constant 1 : index
//CHECK-NEXT:     %6 = "memref.dim"(%arg0, %5) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:     %7 = memref.alloc(%4, %6) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:     scf.for %arg2 = %3 to %0 step %1 {
//CHECK-NEXT:       %8 = arith.constant 8 : index
//CHECK-NEXT:       %9 = arith.constant 1 : index
//CHECK-NEXT:       %10 = "memref.dim"(%arg0, %9) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:       scf.for %arg3 = %3 to %0 step %1 {
//CHECK-NEXT:         %11 = "affine.min"(%1, %0) <{"map" = affine_map<(d0)[s0] -> (8, ((d0 * -1) + s0))>}> : (index, index) -> index
//CHECK-NEXT:         %12 = "memref.dim"(%arg0, %1) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:       }
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
// -----

builtin.module {
  func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
    %0 = arith.constant 8 : index
    %1 = arith.constant 1 : index
    %2 = arith.constant 0 : i32
    %3 = arith.constant 0 : index
    scf.for %arg2 = %3 to %0 step %1{
        scf.for %arg3 = %3 to %0 step %1{
            %8 = "memref.dim"(%arg0, %1) :(memref<?x?xi8, "L3">, index) -> index
            %4 = "affine.min"(%1, %0) <{"map" = affine_map<(d0)[s0] -> (8, ((d0 * -1) + s0))>}> : (index, index) -> index
            %7 = memref.subview %arg1[%arg2, %arg3] [%4, %8] [1, 1] : memref<?x?xi32, "L3"> to memref<?x?xi32, strided<[?, 1], offset: ?>>
            %5 = "memref.dim"(%7, %1) :(memref<?x?xi32, strided<[?, 1], offset: ?>>, index) -> index
            %6 = memref.alloc(%4, %5) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
            }
        }
    func.return
  }
}

// Check if the correct constant is found when dim is called on a subview which itself uses affine.min and a dim operation
//CHECK: builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 8 : index
//CHECK-NEXT:     %1 = arith.constant 1 : index
//CHECK-NEXT:     %2 = arith.constant 0 : i32
//CHECK-NEXT:     %3 = arith.constant 0 : index
//CHECK-NEXT:     %4 = arith.constant 8 : index
//CHECK-NEXT:     %5 = arith.constant 1 : index
//CHECK-NEXT:     %6 = "memref.dim"(%arg1, %5) : (memref<?x?xi32, "L3">, index) -> index
//CHECK-NEXT:     %7 = memref.alloc(%4, %6) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:     scf.for %arg2 = %3 to %0 step %1 {
//CHECK-NEXT:       %8 = arith.constant 8 : index
//CHECK-NEXT:       %9 = arith.constant 1 : index
//CHECK-NEXT:       %10 = "memref.dim"(%arg1, %9) : (memref<?x?xi32, "L3">, index) -> index
//CHECK-NEXT:       scf.for %arg3 = %3 to %0 step %1 {
//CHECK-NEXT:         %11 = "memref.dim"(%arg0, %1) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:         %12 = "affine.min"(%1, %0) <{"map" = affine_map<(d0)[s0] -> (8, ((d0 * -1) + s0))>}> : (index, index) -> index
//CHECK-NEXT:         %13 = memref.subview %arg1[%arg2, %arg3] [%12, %11] [1, 1] : memref<?x?xi32, "L3"> to memref<?x?xi32, strided<[?, 1], offset: ?>>
//CHECK-NEXT:         %14 = "memref.dim"(%13, %1) : (memref<?x?xi32, strided<[?, 1], offset: ?>>, index) -> index
//CHECK-NEXT:       }
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
// -----

builtin.module {
  func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
    %0 = arith.constant 8 : index
    %1 = arith.constant 1 : index
    %2 = arith.constant 0 : i32
    %3 = arith.constant 0 : index
    scf.for %arg2 = %3 to %0 step %1{
        scf.for %arg3 = %3 to %0 step %1{
            %7 = memref.subview %arg1[%arg2, %arg3] [%0, %arg3] [1, 1] : memref<?x?xi32, "L3"> to memref<?x?xi32, strided<[?, 1], offset: ?>>
            %5 = "memref.dim"(%7, %1) :(memref<?x?xi32, strided<[?, 1], offset: ?>>, index) -> index
            %6 = memref.alloc(%0, %5) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
            }
        }
    func.return
  }
}


// Nothing should happen when dim is called on a subview which itself is dependent on a loop variable
//CHECK: builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 8 : index
//CHECK-NEXT:     %1 = arith.constant 1 : index
//CHECK-NEXT:     %2 = arith.constant 0 : i32
//CHECK-NEXT:     %3 = arith.constant 0 : index
//CHECK-NEXT:     scf.for %arg2 = %3 to %0 step %1 {
//CHECK-NEXT:       scf.for %arg3 = %3 to %0 step %1 {
//CHECK-NEXT:         %4 = memref.subview %arg1[%arg2, %arg3] [%0, %arg3] [1, 1] : memref<?x?xi32, "L3"> to memref<?x?xi32, strided<[?, 1], offset: ?>>
//CHECK-NEXT:         %5 = "memref.dim"(%4, %1) : (memref<?x?xi32, strided<[?, 1], offset: ?>>, index) -> index
//CHECK-NEXT:         %6 = memref.alloc(%0, %5) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:       }
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }