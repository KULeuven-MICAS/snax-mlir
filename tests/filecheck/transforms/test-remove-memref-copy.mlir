// RUN: ./compiler/snax-opt %s -p test-remove-memref-copy | filecheck %s

builtin.module {
  %0 = "test.op"() : () -> memref<64xi32, "L3">
  %1 = memref.alloc() {"alignment" = 64 : i64} : memref<64xi32, "L1">
  "memref.copy"(%0, %1) : (memref<64xi32, "L3">, memref<64xi32, "L1">) -> ()
  "test.op"(%1) : (memref<64xi32, "L1">) -> ()
  "memref.copy"(%1, %0) : (memref<64xi32, "L1">, memref<64xi32, "L3">) -> ()
}

// CHECK: builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> memref<64xi32, "L3">
// CHECK-NEXT:   %1 = memref.alloc() {alignment = 64 : i64} : memref<64xi32, "L1">
// CHECK-NEXT:   "test.op"(%1) : (memref<64xi32, "L1">) -> ()
// CHECK-NEXT: }
