// RUN: snax-opt %s -p phs-remove-one-option-switches | filecheck %s

builtin.module {
  phs.pe @acc1 with %0 (%in : i64, %in_1 : i64) {
    %1 = phs.choose @i_i64_i64_o_i64_0 with %0 (%in : i64, %in_1 : i64) -> i64
      0) {
        %2 = arith.addi %in, %in_1 : i64
        phs.yield %2 : i64
      }
    phs.yield %1 : i64
  }

// CHECK:  phs.pe @acc1 (%in : i64, %in_1 : i64) {
// CHECK-NEXT:    %0 = arith.addi %in, %in_1 : i64
// CHECK-NEXT:    phs.yield %0 : i64
// CHECK-NEXT:  }

  phs.pe @acc2 with %0, %1 (%in : i64, %in_1 : i64) {
    %2 = phs.choose @i_i64_i64_o_i64_0 with %0 (%in : i64, %in_1 : i64) -> i64
      0) {
        %3 = arith.addi %in, %in_1 : i64
        phs.yield %3 : i64
      }
    %5 = phs.choose @i_i64_i64_o_i64_1 with %1 (%in : i64, %2 : i64) -> i64
      0) {
        %4 = arith.subi %in, %2 : i64
        phs.yield %4 : i64
      }
    phs.yield %5 : i64
  }
}

// CHECK:   phs.pe @acc2 (%in : i64, %in_1 : i64) {
// CHECK-NEXT:      %0 = arith.addi %in, %in_1 : i64
// CHECK-NEXT:      %1 = arith.subi %in, %0 : i64
// CHECK-NEXT:      phs.yield %1 : i64
// CHECK-NEXT:   }
