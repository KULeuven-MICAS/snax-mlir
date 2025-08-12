// RUN: snax-opt -p construct-pipeline %s | filecheck %s

%0 = memref.alloc() : memref<1x2xi8>
%1 = memref.alloc() : memref<1x2xi8>
%2 = memref.alloc() : memref<1x2xi8>
%3 = memref.alloc() : memref<1x2xi8>
%lb = arith.constant 0 : index
%ub = arith.constant 10 : index
%step = arith.constant 1 : index
scf.for %i = %lb to %ub step %step {
  "test.op"(%i) : (index) -> ()
  "memref.copy"(%0, %1) : (memref<1x2xi8>, memref<1x2xi8>) -> ()
  "snax.cluster_sync_op"() : () -> ()
  "dart.operation"(%1, %2) <{patterns = [], operandSegmentSizes = array<i32: 1, 1>}> ({
    dart.yield
  }) : (memref<1x2xi8>, memref<1x2xi8>) -> ()
  "snax.cluster_sync_op"() : () -> ()
  "memref.copy"(%2, %3) : (memref<1x2xi8>, memref<1x2xi8>) -> ()
  "snax.cluster_sync_op"() : () -> ()
}

// CHECK:      %0 = memref.alloc() : memref<1x2xi8>
// CHECK-NEXT: %1 = memref.alloc() : memref<1x2xi8>
// CHECK-NEXT: %2 = memref.alloc() : memref<1x2xi8>
// CHECK-NEXT: %3 = memref.alloc() : memref<1x2xi8>
// CHECK-NEXT: %lb = arith.constant 0 : index
// CHECK-NEXT: %ub = arith.constant 10 : index
// CHECK-NEXT: %step = arith.constant 1 : index
// CHECK-NEXT: scf.for %i = %lb to %ub step %step {
// CHECK-NEXT:   pipeline.pipeline {
// CHECK-NEXT:     pipeline.index %i -> {
// CHECK-NEXT:     ^bb0(%i_1 : index):
// CHECK-NEXT:       "test.op"(%i_1) : (index) -> ()
// CHECK-NEXT:       pipeline.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     pipeline.stage 0 ins(%0 : memref<1x2xi8>) outs(%1 : memref<1x2xi8>) {
// CHECK-NEXT:     ^bb1(%4 : memref<1x2xi8>, %5 : memref<1x2xi8>):
// CHECK-NEXT:       "memref.copy"(%4, %5) : (memref<1x2xi8>, memref<1x2xi8>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     pipeline.stage 1 ins(%1 : memref<1x2xi8>) outs(%2 : memref<1x2xi8>) {
// CHECK-NEXT:     ^bb2(%6 : memref<1x2xi8>, %7 : memref<1x2xi8>):
// CHECK-NEXT:       "dart.operation"(%6, %7) <{patterns = [], operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:         dart.yield
// CHECK-NEXT:       }) : (memref<1x2xi8>, memref<1x2xi8>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     pipeline.stage 2 ins(%2 : memref<1x2xi8>) outs(%3 : memref<1x2xi8>) {
// CHECK-NEXT:     ^bb3(%8 : memref<1x2xi8>, %9 : memref<1x2xi8>):
// CHECK-NEXT:       "memref.copy"(%8, %9) : (memref<1x2xi8>, memref<1x2xi8>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
