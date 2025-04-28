// RUN: snax-opt --split-input-file -p pipeline-canonicalize-for %s | filecheck %s

%lb = arith.constant 0 : index
%ub = arith.constant 10 : index
%step = arith.constant 1 : index
scf.for %i = %lb to %ub step %step {
  "test.op"(%i) : (index) -> ()
}

//               nothing changes:
// CHECK:        %lb = arith.constant 0 : index
// CHECK-NEXT:   %ub = arith.constant 10 : index
// CHECK-NEXT:   %step = arith.constant 1 : index
// CHECK-NEXT:   scf.for %i = %lb to %ub step %step {
// CHECK-NEXT:     "test.op"(%i) : (index) -> ()
// CHECK-NEXT:   }

// -----

%lb = arith.constant 0 : index
%ub = arith.constant 10 : index
%step = arith.constant 2 : index
scf.for %i = %lb to %ub step %step {
  "test.op"(%i) : (index) -> ()
}

//               step is rewritten to 1:
// CHECK:        %lb = arith.constant 0 : index
// CHECK-NEXT:   %ub = arith.constant 10 : index
// CHECK-NEXT:   %step = arith.constant 2 : index
// CHECK-NEXT:   %0 = arith.constant 1 : index
// CHECK-NEXT:   %1 = arith.constant 5 : index
// CHECK-NEXT:   scf.for %i = %lb to %1 step %0 {
// CHECK-NEXT:     %i_1 = arith.muli %step, %i : index
// CHECK-NEXT:     "test.op"(%i_1) : (index) -> ()
// CHECK-NEXT:   }


