// RUN: snax-opt %s -p convert-phs-to-hw | firtool --format=mlir | filecheck %s

phs.pe @myfirstaccelerator with %0, %1, %2, %3, %4, %5 (%6 : i32, %7 : i32) {
  %8 = phs.choose @_0 with %0 (%6 : i32, %7 : i32) -> i32
    0) {
      %9 = arith.muli %6, %7 : i32
      phs.yield %9 : i32
    }
    1) {
      %10 = arith.addi %6, %7 : i32
      phs.yield %10 : i32
    }
    2) {
      %11 = arith.subi %6, %7 : i32
      phs.yield %11 : i32
    }
  %12 = phs.choose @_1 with %1 (%6 : i32, %8 : i32) -> i32
    0) {
      %13 = arith.muli %6, %8 : i32
      phs.yield %13 : i32
    }
    1) {
      %14 = arith.addi %6, %8 : i32
      phs.yield %14 : i32
    }
  %15 = phs.mux with %2 (%8 : i32, %12 : i32) -> i32
  %16 = phs.mux with %5 (%12 : i32, %7 : i32) -> i32
  %17 = phs.choose @_2 with %3 (%8 : i32, %16 : i32) -> i32
    0) {
      %18 = arith.muli %8, %12 : i32
      phs.yield %18 : i32
    }
    1) {
      %19 = arith.divui %8, %16 : i32
      phs.yield %19 : i32
    }
  %20 = phs.mux with %4 (%15 : i32, %17 : i32) -> i32
  phs.yield %20 : i32
}

// CHECK: module myfirstaccelerator(      // <stdin>:2:3
// CHECK-NEXT:  input  [31:0] data_0, // <stdin>:2:36
// CHECK-NEXT:                data_1, // <stdin>:2:58
// CHECK-NEXT:  input  [1:0]  switch_0,       // <stdin>:2:80
// CHECK-NEXT:  input         switch_1,       // <stdin>:2:103
// CHECK-NEXT:                switch_2,       // <stdin>:2:126
// CHECK-NEXT:                switch_3,       // <stdin>:2:149
// CHECK-NEXT:                switch_4,       // <stdin>:2:172
// CHECK-NEXT:                switch_5,       // <stdin>:2:195
// CHECK-NEXT:  output [31:0] out_0   // <stdin>:2:219
// CHECK-NEXT:);
//
// CHECK:  wire [2:0][31:0] _GEN = {{[{][{]}}data_0 * data_1}, {data_0 + data_1}, {data_0 - data_1{{[}][}]}};   // <stdin>:3:10, :4:10, :5:10, :6:10
// CHECK-NEXT:  wire [1:0][31:0] _GEN_0 = {{[{][{]}}data_0 * _GEN[switch_0]}, {data_0 + _GEN[switch_0]{{[}][}]}};    // <stdin>:6:10, :7:10, :8:10, :9:10, :10:10
// CHECK-NEXT:  wire [1:0][31:0] _GEN_1 =
// CHECK-NEXT:    {{[{][{]}}_GEN[switch_0] * _GEN_0[switch_1]},
// CHECK-NEXT:     {_GEN[switch_0] / (switch_5 ? data_1 : _GEN_0[switch_1]){{[}][}]}};        // <stdin>:6:10, :7:10, :10:10, :11:10, :13:11, :14:11, :15:11, :16:11
// CHECK-NEXT:  assign out_0 =
// CHECK-NEXT:    switch_4 ? _GEN_1[switch_3] : switch_2 ? _GEN_0[switch_1] : _GEN[switch_0]; // <stdin>:6:10, :7:10, :10:10, :11:10, :12:10, :16:11, :17:11, :18:11, :19:5
// CHECK-NEXT:endmodule
