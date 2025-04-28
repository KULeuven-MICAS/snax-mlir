// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP


builtin.module {
  %0 = memref.alloc() : memref<1x2xi32>
  %1 = memref.alloc() : memref<1x2xi32>
  %lb = arith.constant 0 : index
  %ub = arith.constant 10 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    pipeline.pipeline {
      %2, %3 = pipeline.index %i -> i32, memref<1x2xi32> {
      ^0(%4 : index):
        %5 = arith.constant 0 : i32
        %6 = memref.alloc() : memref<1x2xi32>
        pipeline.yield %5, %6 : i32, memref<1x2xi32>
      }
      pipeline.stage 0 {
        "memref.copy"(%3, %3) : (memref<1x2xi32>, memref<1x2xi32>) -> ()
      }
      pipeline.stage 1 ins(%0 : memref<1x2xi32>) outs(%1 : memref<1x2xi32>) {
      ^1(%7 : memref<1x2xi32>, %8 : memref<1x2xi32>):
        "memref.copy"(%7, %8) : (memref<1x2xi32>, memref<1x2xi32>) -> ()
      }
    }
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = memref.alloc() : memref<1x2xi32>
// CHECK-NEXT:   %1 = memref.alloc() : memref<1x2xi32>
// CHECK-NEXT:   %lb = arith.constant 0 : index
// CHECK-NEXT:   %ub = arith.constant 10 : index
// CHECK-NEXT:   %step = arith.constant 1 : index
// CHECK-NEXT:   scf.for %i = %lb to %ub step %step {
// CHECK-NEXT:     pipeline.pipeline {
// CHECK-NEXT:       %2, %3 = pipeline.index %i -> i32, memref<1x2xi32> {
// CHECK-NEXT:       ^0(%4 : index):
// CHECK-NEXT:         %5 = arith.constant 0 : i32
// CHECK-NEXT:         %6 = memref.alloc() : memref<1x2xi32>
// CHECK-NEXT:         pipeline.yield %5, %6 : i32, memref<1x2xi32>
// CHECK-NEXT:       }
// CHECK-NEXT:       pipeline.stage 0 {
// CHECK-NEXT:         "memref.copy"(%3, %3) : (memref<1x2xi32>, memref<1x2xi32>) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:       pipeline.stage 1 ins(%0 : memref<1x2xi32>) outs(%1 : memref<1x2xi32>) {
// CHECK-NEXT:       ^1(%7 : memref<1x2xi32>, %8 : memref<1x2xi32>):
// CHECK-NEXT:         "memref.copy"(%7, %8) : (memref<1x2xi32>, memref<1x2xi32>) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK-GENERIC:      "builtin.module"() ({
// CHECK-GENERIC-NEXT:   %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x2xi32>
// CHECK-GENERIC-NEXT:   %1 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x2xi32>
// CHECK-GENERIC-NEXT:   %lb = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK-GENERIC-NEXT:   %ub = "arith.constant"() <{value = 10 : index}> : () -> index
// CHECK-GENERIC-NEXT:   %step = "arith.constant"() <{value = 1 : index}> : () -> index
// CHECK-GENERIC-NEXT:   "scf.for"(%lb, %ub, %step) ({
// CHECK-GENERIC-NEXT:   ^0(%i : index):
// CHECK-GENERIC-NEXT:     "pipeline.pipeline"() ({
// CHECK-GENERIC-NEXT:       %2, %3 = "pipeline.index"(%i) ({
// CHECK-GENERIC-NEXT:       ^1(%4 : index):
// CHECK-GENERIC-NEXT:         %5 = "arith.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-GENERIC-NEXT:         %6 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x2xi32>
// CHECK-GENERIC-NEXT:         "pipeline.yield"(%5, %6) : (i32, memref<1x2xi32>) -> ()
// CHECK-GENERIC-NEXT:       }) : (index) -> (i32, memref<1x2xi32>)
// CHECK-GENERIC-NEXT:       "pipeline.stage"() <{index = 0 : index, operandSegmentSizes = array<i32: 0, 0>}> ({
// CHECK-GENERIC-NEXT:         "memref.copy"(%3, %3) : (memref<1x2xi32>, memref<1x2xi32>) -> ()
// CHECK-GENERIC-NEXT:       }) : () -> ()
// CHECK-GENERIC-NEXT:       "pipeline.stage"(%0, %1) <{index = 1 : index, operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:       ^2(%7 : memref<1x2xi32>, %8 : memref<1x2xi32>):
// CHECK-GENERIC-NEXT:         "memref.copy"(%7, %8) : (memref<1x2xi32>, memref<1x2xi32>) -> ()
// CHECK-GENERIC-NEXT:       }) : (memref<1x2xi32>, memref<1x2xi32>) -> ()
// CHECK-GENERIC-NEXT:     }) : () -> ()
// CHECK-GENERIC-NEXT:     "scf.yield"() : () -> ()
// CHECK-GENERIC-NEXT:   }) : (index, index, index) -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()
