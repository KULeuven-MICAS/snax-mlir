%lb = arith.constant 0 : index
%ub = arith.constant 10 : index
%step = arith.constant 1 : index
scf.for %i = %lb to %ub step %step {
  pipeline.pipeline {
    %0 = pipeline.index %i -> index {
    ^0(%1 : index):
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


