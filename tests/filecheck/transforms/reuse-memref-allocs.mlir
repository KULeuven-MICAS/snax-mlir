// RUN: snax-opt --split-input-file %s -p reuse-memref-allocs | filecheck %s

builtin.module {
  func.func @streamer_matmul_0(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
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
//CHECK-NEXT:   func.func @streamer_matmul_0(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 8 : index
//                dead code elimination:
//                 vvvvvvvvvvvvvvvvvv
//                %1 = arith.constant 1 : index
//                %2 = arith.constant 0 : i32
//                %3 = arith.constant 0 : index
//CHECK-NEXT:     %1 = memref.alloc(%0, %0) {alignment = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//                dead code elimination:
//                 vvvvvvvvvvvvvvvvvv
//                scf.for %arg2 = %3 to %0 step %1 {
//                }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
// -----


builtin.module {
  func.func @streamer_matmul_1(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
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
//CHECK-NEXT:   func.func @streamer_matmul_1(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//                dead code elimination
//                 vvvvvvvvvvvvvvvvvv
//                %0 = arith.constant 8 : index
//                %1 = arith.constant 1 : index
//                %2 = arith.constant 0 : i32
//                %3 = arith.constant 0 : index
//CHECK-NEXT:     %0 = arith.constant 16 : index
//CHECK-NEXT:     %1 = arith.constant 8 : index
//CHECK-NEXT:     %2 = memref.alloc(%0, %1) {alignment = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//                dead code elimination
//                 vvvvvvvvvvvvvvvvvv
//                scf.for %arg2 = %3 to %0 step %1 {
//                }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
// -----

builtin.module {
  func.func @streamer_matmul_2(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
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
//CHECK-NEXT:   func.func @streamer_matmul_2(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 8 : index
//CHECK-NEXT:     %1 = arith.constant 1 : index
//                dead code elimination
//                 vvvvvvvvvvvvvvvvvv
//                %2 = arith.constant 0 : i32
//CHECK-NEXT:     %2 = arith.constant 0 : index
//CHECK-NEXT:     %3 = arith.constant 8 : index
//CHECK-NEXT:     scf.for %arg2 = %2 to %0 step %1 {
//CHECK-NEXT:       %4 = memref.alloc(%arg2, %3) {alignment = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
// -----

builtin.module {
  func.func @streamer_matmul_3(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
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
//CHECK-NEXT:   func.func @streamer_matmul_3(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//                dead code elimination
//                 vvvvvvvvvvvvvvvvvv
//                %0 = arith.constant 8 : index
//CHECK-NEXT:     %0 = arith.constant 1 : index
//                dead code elimination
//                 vvvvvvvvvvvvvvvvvv
//                %2 = arith.constant 0 : i32
//CHECK-NEXT:     %1 = arith.constant 0 : index
//CHECK-NEXT:     %2 = "memref.dim"(%arg0, %1) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:     %3 = "memref.dim"(%arg0, %0) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:     %4 = memref.alloc(%2, %3) {alignment = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//                dead code elimination
//                 vvvvvvvvvvvvvvvvvv
//                scf.for %arg2 = %3 to %0 step %1 {
//                }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
// -----

builtin.module {
  func.func @streamer_matmul_4(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
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
//CHECK-NEXT:  func.func @streamer_matmul_4(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//               dead code elimination
//                vvvvvvvvvvvvvvvvvv
//               %0 = arith.constant 8 : index
//CHECK-NEXT:    %0 = arith.constant 1 : index
//               dead code elimination
//                vvvvvvvvvvvvvvvvvv
//               %2 = arith.constant 0 : i32
//CHECK-NEXT:    %1 = arith.constant 0 : index
//CHECK-NEXT:    %2 = "memref.dim"(%arg0, %1) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:    %3 = "memref.dim"(%arg0, %0) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:    %4 = memref.alloc(%2, %3) {alignment = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//               dead code elimination
//                vvvvvvvvvvvvvvvvvv
//               scf.for %arg2 = %3 to %0 step %1 {
//               }
//CHECK-NEXT:    func.return
//CHECK-NEXT:  }
//CHECK-NEXT:}
// -----

builtin.module {
  func.func @streamer_matmul_5(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
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

// Nothing should happen if no dim operation is used with a subview to directly dependant on the affine.min
// Only the dim operation can be lowered here
//CHECK: builtin.module {
//CHECK-NEXT:    func.func @streamer_matmul_5(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//CHECK-NEXT:    %0 = arith.constant 8 : index
//CHECK-NEXT:    %1 = arith.constant 1 : index
//               dead code elimination
//                vvvvvvvvvvvvvvvvvv
//               %2 = arith.constant 0 : i32
//CHECK-NEXT:    %2 = arith.constant 0 : index
//CHECK-NEXT:    %3 = "memref.dim"(%arg0, %1) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:    scf.for %arg2 = %2 to %0 step %1 {
//CHECK-NEXT:      scf.for %arg3 = %2 to %0 step %1 {
//CHECK-NEXT:        %4 = "affine.min"(%1, %0) <{map = affine_map<(d0)[s0] -> (8, ((d0 * -1) + s0))>}> : (index, index) -> index
//CHECK-NEXT:        %5 = memref.alloc(%4, %3) {alignment = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:      }
//CHECK-NEXT:    }
//CHECK-NEXT:    func.return
//CHECK-NEXT:  }
//CHECK-NEXT:}
// -----

builtin.module {
  func.func @streamer_matmul_6(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
    %0 = arith.constant 8 : index
    %1 = arith.constant 1 : index
    %2 = arith.constant 0 : i32
    %3 = arith.constant 0 : index
    scf.for %arg2 = %3 to %0 step %1{
        scf.for %arg3 = %3 to %0 step %1{
            %8 = "memref.dim"(%arg0, %1) :(memref<?x?xi8, "L3">, index) -> index
            %4 = "affine.min"(%1, %0) <{"map" = affine_map<(d0)[s0] -> (8, ((d0 * -1) + s0))>}> : (index, index) -> index
            %7 = memref.subview %arg1[%arg2, %arg3] [%4, %8] [1, 1] : memref<?x?xi32, "L3"> to memref<?x?xi32, strided<[?, 1], offset: ?>>
            "test.op"(%7) : (memref<?x?xi32, strided<[?, 1], offset: ?>>) -> ()
            %5 = "memref.dim"(%7, %3) :(memref<?x?xi32, strided<[?, 1], offset: ?>>, index) -> index
            %6 = memref.alloc(%4, %5) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
            }
        }
    func.return
  }
}

// Check if the correct constant is found when dim is called on a subview which itself uses affine.min and a dim operation
//CHECK: builtin.module {
//CHECK-NEXT:  func.func @streamer_matmul_6(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//CHECK-NEXT:    %0 = arith.constant 8 : index
//CHECK-NEXT:    %1 = arith.constant 1 : index
//               dead code eliminiation
//                vvvvvvvvvvvvvvvvvv
//               %2 = arith.constant 0 : i32
//CHECK-NEXT:    %2 = arith.constant 0 : index
//CHECK-NEXT:    %3 = "memref.dim"(%arg0, %1) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:    %4 = arith.constant 8 : index
//CHECK-NEXT:    %5 = memref.alloc(%4, %4) {alignment = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:    scf.for %arg2 = %2 to %0 step %1 {
//CHECK-NEXT:      scf.for %arg3 = %2 to %0 step %1 {
//CHECK-NEXT:        %6 = "affine.min"(%1, %0) <{map = affine_map<(d0)[s0] -> (8, ((d0 * -1) + s0))>}> : (index, index) -> index
//CHECK-NEXT:        %7 = memref.subview %arg1[%arg2, %arg3] [%4, %3] [1, 1] : memref<?x?xi32, "L3"> to memref<?x?xi32, strided<[?, 1], offset: ?>>
//CHECK-NEXT:        "test.op"(%7) : (memref<?x?xi32, strided<[?, 1], offset: ?>>) -> ()
//CHECK-NEXT:      }
//CHECK-NEXT:    }
//CHECK-NEXT:    func.return
//CHECK-NEXT:  }
//CHECK-NEXT:}
// -----

builtin.module {
  func.func @streamer_matmul_7(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
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
//CHECK-NEXT:   func.func @streamer_matmul_7(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 8 : index
//CHECK-NEXT:     %1 = arith.constant 1 : index
//                dead code elimination
//                 vvvvvvvvvvvvvvvvvv
//                %2 = arith.constant 0 : i32
//CHECK-NEXT:     %2 = arith.constant 0 : index
//CHECK-NEXT:     scf.for %arg2 = %2 to %0 step %1 {
//CHECK-NEXT:       scf.for %arg3 = %2 to %0 step %1 {
//CHECK-NEXT:         %3 = memref.subview %arg1[%arg2, %arg3] [%0, %arg3] [1, 1] : memref<?x?xi32, "L3"> to memref<?x?xi32, strided<[?, 1], offset: ?>>
//CHECK-NEXT:         %4 = "memref.dim"(%3, %1) : (memref<?x?xi32, strided<[?, 1], offset: ?>>, index) -> index
//CHECK-NEXT:         %5 = memref.alloc(%0, %4) {alignment = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:       }
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
// -----

builtin.module {
  func.func @streamer_matmul_8(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
    %0 = arith.constant 8 : index
    %1 = arith.constant 1 : index
    %2 = arith.constant 0 : i32
    %3 = arith.constant 0 : index
    scf.for %arg2 = %3 to %0 step %1{
        %4 = "memref.dim"(%arg0, %3) :(memref<?x?xi8, "L3">, index) -> index
        %5 = "memref.dim"(%arg0, %1) :(memref<?x?xi8, "L3">, index) -> index
        %6 = memref.alloc(%4, %5) {"alignment" = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
        %7 = arith.addi %4, %5 : index
        "test.op"(%7) : (index) -> ()
        }
    func.return
  }
}

// Nothing should happen when the Dim result is used by operations other than subviews and allocs
//CHECK: builtin.module {
//CHECK-NEXT:   func.func @streamer_matmul_8(%arg0 : memref<?x?xi8, "L3">, %arg1 : memref<?x?xi32, "L3">) {
//CHECK-NEXT:     %0 = arith.constant 8 : index
//CHECK-NEXT:     %1 = arith.constant 1 : index
//                dead code elimination
//                 vvvvvvvvvvvvvvvvvv
//                %2 = arith.constant 0 : i32
//CHECK-NEXT:     %2 = arith.constant 0 : index
//CHECK-NEXT:     scf.for %arg2 = %2 to %0 step %1 {
//CHECK-NEXT:       %3 = "memref.dim"(%arg0, %2) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:       %4 = "memref.dim"(%arg0, %1) : (memref<?x?xi8, "L3">, index) -> index
//CHECK-NEXT:       %5 = memref.alloc(%3, %4) {alignment = 64 : i64} : memref<?x?xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>, "L1">
//CHECK-NEXT:       %6 = arith.addi %3, %4 : index
//CHECK-NEXT:       "test.op"(%6) : (index) -> ()
//CHECK-NEXT:     }
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }
