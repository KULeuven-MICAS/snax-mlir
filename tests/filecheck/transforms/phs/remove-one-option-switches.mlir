// RUN: snax-opt %s -p phs-remove-one-option-switches | filecheck %s

builtin.module {
  phs.pe @acc1 with %0 (%in : i64, %in_1 : i64, %out : i64) {
    %1 = phs.choose @i_i64_i64_o_i64_0 with %0 (%in : i64, %in_1 : i64) -> i64
      0) {
        %2 = arith.addi %in, %in_1 : i64
        phs.yield %2 : i64
      }
    phs.yield %1 : i64
  }
}

// CHECK: builtin.module {
// CHECK-NEXT:  phs.pe @acc1 (%in : i64, %in_1 : i64, %out : i64) {
// CHECK-NEXT:    %0 = arith.addi %in, %in_1 : i64
// CHECK-NEXT:    phs.yield %0 : i64
// CHECK-NEXT:  }
// CHECK-NEXT:}
