// RUN: snax-opt %s -p unroll-pipeline --split-input-file | filecheck %s

%lb = arith.constant 0 : index
%ub = arith.constant 10 : index
%step = arith.constant 1 : index
scf.for %i = %lb to %ub step %step {
  pipeline.pipeline {
    %0 = pipeline.index %i -> index {
    ^bb0(%1 : index):
      pipeline.yield %1 : index
    }
    pipeline.stage 0 {
      "test.op"(%0) {hello} : (index) -> ()
    }
    pipeline.stage 1 {
      "test.op"(%0) {world} : (index) -> ()
    }
  }
}

// CHECK:      %lb = arith.constant 0 : index
// CHECK-NEXT: %ub = arith.constant 10 : index
// CHECK-NEXT: %step = arith.constant 1 : index
// CHECK-NEXT: %0 = arith.constant 0 : index
// CHECK-NEXT: "test.op"(%0) {hello} : (index) -> ()
// CHECK-NEXT: "snax.cluster_sync_op"() : () -> ()
// CHECK-NEXT: %lb_1 = arith.constant 1 : index
// CHECK-NEXT: scf.for %i = %lb_1 to %ub step %step {
// CHECK-NEXT:   %1 = arith.constant 1 : index
// CHECK-NEXT:   %2 = arith.subi %i, %1 : index
// CHECK-NEXT:   "test.op"(%i) {hello} : (index) -> ()
// CHECK-NEXT:   "test.op"(%2) {world} : (index) -> ()
// CHECK-NEXT:   "snax.cluster_sync_op"() : () -> ()
// CHECK-NEXT: }
// CHECK-NEXT: %3 = arith.constant 1 : index
// CHECK-NEXT: %4 = arith.subi %ub, %3 : index
// CHECK-NEXT: "test.op"(%4) {world} : (index) -> ()
// CHECK-NEXT: "snax.cluster_sync_op"() : () -> ()

// -----

%lb = arith.constant 0 : index
%ub = arith.constant 10 : index
%step = arith.constant 1 : index
scf.for %i = %lb to %ub step %step {
  pipeline.pipeline {
    %0 = pipeline.index %i -> index {
    ^bb0(%1 : index):
      pipeline.yield %1 : index
    }
    pipeline.stage 0 {
      "test.op"(%0) {yer} : (index) -> ()
    }
    pipeline.stage 1 {
      "test.op"(%0) {a} : (index) -> ()
    }
    pipeline.stage 2 {
      "test.op"(%0) {wizard} : (index) -> ()
    }
  }
}

// CHECK:      %lb = arith.constant 0 : index
// CHECK-NEXT  %ub = arith.constant 10 : index
// CHECK-NEXT  %step = arith.constant 1 : index
// CHECK-NEXT  %0 = arith.constant 0 : index
// CHECK-NEXT  "test.op"(%0) {yer} : (index) -> ()
// CHECK-NEXT  "snax.cluster_sync_op"() : () -> ()
// CHECK-NEXT  %1 = arith.constant 1 : index
// CHECK-NEXT  "test.op"(%1) {yer} : (index) -> ()
// CHECK-NEXT  "test.op"(%0) {a} : (index) -> ()
// CHECK-NEXT  "snax.cluster_sync_op"() : () -> ()
// CHECK-NEXT  %lb_1 = arith.constant 2 : index
// CHECK-NEXT  scf.for %i = %lb_1 to %ub step %step {
// CHECK-NEXT    %2 = arith.constant 2 : index
// CHECK-NEXT    %3 = arith.subi %i, %2 : index
// CHECK-NEXT    %4 = arith.constant 1 : index
// CHECK-NEXT    %5 = arith.subi %i, %4 : index
// CHECK-NEXT    "test.op"(%i) {yer} : (index) -> ()
// CHECK-NEXT    "test.op"(%5) {a} : (index) -> ()
// CHECK-NEXT    "test.op"(%3) {wizard} : (index) -> ()
// CHECK-NEXT    "snax.cluster_sync_op"() : () -> ()
// CHECK-NEXT  }
// CHECK-NEXT  %6 = arith.constant 1 : index
// CHECK-NEXT  %7 = arith.subi %ub, %6 : index
// CHECK-NEXT  %8 = arith.constant 2 : index
// CHECK-NEXT  %9 = arith.subi %ub, %8 : index
// CHECK-NEXT  "test.op"(%7) {a} : (index) -> ()
// CHECK-NEXT  "test.op"(%9) {wizard} : (index) -> ()
// CHECK-NEXT  "snax.cluster_sync_op"() : () -> ()
// CHECK-NEXT  "test.op"(%7) {wizard} : (index) -> ()
// CHECK-NEXT  "snax.cluster_sync_op"() : () -> ()
