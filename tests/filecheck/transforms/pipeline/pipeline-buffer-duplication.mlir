%0 = memref.alloc() : memref<1x2xi8>
%1 = memref.alloc() : memref<1x2xi8>
%2 = memref.alloc() : memref<1x2xi8>
%3 = memref.alloc() : memref<1x2xi8>
%lb = arith.constant 0 : index
%ub = arith.constant 10 : index
%step = arith.constant 1 : index
scf.for %i = %lb to %ub step %step {
  pipeline.pipeline {
    pipeline.index %i -> {
    ^0(%4 : index):
      pipeline.yield
    }
    pipeline.stage 0 ins(%0 : memref<1x2xi8>) outs(%1 : memref<1x2xi8>) {
    ^1(%5 : memref<1x2xi8>, %6 : memref<1x2xi8>):
      "test.op"(%5, %6) : (memref<1x2xi8>, memref<1x2xi8>) -> ()
    }
    pipeline.stage 1 ins(%1 : memref<1x2xi8>) outs(%2 : memref<1x2xi8>) {
    ^2(%7 : memref<1x2xi8>, %8 : memref<1x2xi8>):
      "test.op"(%7, %8) : (memref<1x2xi8>, memref<1x2xi8>) -> ()
    }
    pipeline.stage 2 ins(%2 : memref<1x2xi8>) outs(%3 : memref<1x2xi8>) {
    ^3(%9 : memref<1x2xi8>, %10 : memref<1x2xi8>):
      "test.op"(%9, %10) : (memref<1x2xi8>, memref<1x2xi8>) -> ()
    }
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = memref.alloc() : memref<1x2xi8>
// CHECK-NEXT:   %1 = memref.alloc() : memref<1x2xi8>
// CHECK-NEXT:   %2 = memref.alloc() : memref<1x2xi8>
// CHECK-NEXT:   %3 = memref.alloc() : memref<1x2xi8>
// CHECK-NEXT:   %4 = memref.alloc() : memref<1x2xi8>
// CHECK-NEXT:   %5 = memref.alloc() : memref<1x2xi8>
// CHECK-NEXT:   %lb = arith.constant 0 : index
// CHECK-NEXT:   %ub = arith.constant 10 : index
// CHECK-NEXT:   %step = arith.constant 1 : index
// CHECK-NEXT:   scf.for %i = %lb to %ub step %step {
// CHECK-NEXT:     pipeline.pipeline {
// CHECK-NEXT:       %6, %7, %8, %9 = pipeline.index %i -> memref<1x2xi8>, memref<1x2xi8>, memref<1x2xi8>, memref<1x2xi8> {
// CHECK-NEXT:       ^0(%10 : index):
// CHECK-NEXT:         %11 = arith.constant 2 : index
// CHECK-NEXT:         %12 = arith.constant 0 : index
// CHECK-NEXT:         %13 = arith.remui %10, %11 : index
// CHECK-NEXT:         %14 = arith.cmpi eq, %12, %13 : index
// CHECK-NEXT:         %15 = arith.select %14, %1, %2 : memref<1x2xi8>
// CHECK-NEXT:         %16 = arith.constant 2 : index
// CHECK-NEXT:         %17 = arith.constant 0 : index
// CHECK-NEXT:         %18 = arith.remui %10, %16 : index
// CHECK-NEXT:         %19 = arith.cmpi eq, %17, %18 : index
// CHECK-NEXT:         %20 = arith.select %19, %3, %4 : memref<1x2xi8>
// CHECK-NEXT:         pipeline.yield %0, %15, %20, %5 : memref<1x2xi8>, memref<1x2xi8>, memref<1x2xi8>, memref<1x2xi8>
// CHECK-NEXT:       }
// CHECK-NEXT:       pipeline.stage 0 {
// CHECK-NEXT:         "test.op"(%6, %7) : (memref<1x2xi8>, memref<1x2xi8>) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:       pipeline.stage 1 {
// CHECK-NEXT:         "test.op"(%7, %8) : (memref<1x2xi8>, memref<1x2xi8>) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:       pipeline.stage 2 {
// CHECK-NEXT:         "test.op"(%8, %9) : (memref<1x2xi8>, memref<1x2xi8>) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
