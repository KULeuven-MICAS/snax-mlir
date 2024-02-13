// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_SINGLETRIP

"builtin.module"() ({
  %0 = "test.op"() : () -> memref<8x8xi32, strided<[1, 8]>, 1 : i32>
  %1 = "snax.layout_cast"(%0) : (memref<8x8xi32, strided<[1, 8]>, 1 : i32>) -> memref<8x8xi32, strided<[1, 16]>, 1 : i32>
}) : () -> ()


// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "test.op"() : () -> memref<8x8xi32, strided<[1, 8]>, 1 : i32>
// CHECK-NEXT:   %1 = "snax.layout_cast"(%0) : (memref<8x8xi32, strided<[1, 8]>, 1 : i32>) -> memref<8x8xi32, strided<[1, 16]>, 1 : i32>
// CHECK-NEXT: }) : () -> ()
