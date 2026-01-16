// RUN: snax-opt %s -p phs-dead-input-removal | filecheck %s

builtin.module {
  phs.pe @acc1 with %0 (%in : i64, %in_1 : i64, %out : i64) {
    %1 = phs.choose @i_i64_i64_o_i64_0 with %0 (%in : i64, %in_1 : i64) -> i64
      0) {
        %2 = arith.addi %in, %in_1 : i64
        phs.yield %2 : i64
      }
      1) {
        %3 = arith.subi %in, %in_1 : i64
        phs.yield %3 : i64
      }
      2) {
        %4 = arith.muli %in, %in_1 : i64
        phs.yield %4 : i64
      }
      3) {
        %5 = arith.xori %in, %in_1 : i64
        phs.yield %5 : i64
      }
    phs.yield %1 : i64
  }
}

// CHECK: builtin.module {
// CHECK-NEXT:   phs.pe @acc1 with %0 (%in : i64, %in_1 : i64) {
// CHECK-NEXT:     %1 = phs.choose @i_i64_i64_o_i64_0 with %0 (%in : i64, %in_1 : i64) -> i64
// CHECK-NEXT:       0) {
// CHECK-NEXT:         %2 = arith.addi %in, %in_1 : i64
// CHECK-NEXT:         phs.yield %2 : i64
// CHECK-NEXT:       }
// CHECK-NEXT:       1) {
// CHECK-NEXT:         %3 = arith.subi %in, %in_1 : i64
// CHECK-NEXT:         phs.yield %3 : i64
// CHECK-NEXT:       }
// CHECK-NEXT:       2) {
// CHECK-NEXT:         %4 = arith.muli %in, %in_1 : i64
// CHECK-NEXT:         phs.yield %4 : i64
// CHECK-NEXT:       }
// CHECK-NEXT:       3) {
// CHECK-NEXT:         %5 = arith.xori %in, %in_1 : i64
// CHECK-NEXT:         phs.yield %5 : i64
// CHECK-NEXT:       }
// CHECK-NEXT:     phs.yield %1 : i64
// CHECK-NEXT:   }
// CHECK-NEXT: }
