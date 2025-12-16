// RUN: snax-opt %s -p convert-phs-to-hw | firtool --format=mlir | filecheck %s --check-prefix=SV
// RUN: snax-opt %s -p convert-phs-to-hw | filecheck %s

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

// SV: module myfirstaccelerator(
// SV-NEXT:  input  [31:0] data_0,
// SV-NEXT:                data_1,
// SV-NEXT:  input  [1:0]  switch_0,
// SV-NEXT:  input         switch_1,
// SV-NEXT:                switch_2,
// SV-NEXT:                switch_3,
// SV-NEXT:                switch_4,
// SV-NEXT:                switch_5,
// SV-NEXT:  output [31:0] out_0
// SV-NEXT:);
//
// SV:  wire [2:0][31:0] _GEN = {{[{][{]}}data_0 * data_1}, {data_0 + data_1}, {data_0 - data_1{{[}][}]}};
// SV-NEXT:  wire [1:0][31:0] _GEN_0 = {{[{][{]}}data_0 * _GEN[switch_0]}, {data_0 + _GEN[switch_0]{{[}][}]}};
// SV-NEXT:  wire [1:0][31:0] _GEN_1 =
// SV-NEXT:    {{[{][{]}}_GEN[switch_0] * _GEN_0[switch_1]},
// SV-NEXT:     {_GEN[switch_0] / (switch_5 ? data_1 : _GEN_0[switch_1]){{[}][}]}};
// SV-NEXT:  assign out_0 =
// SV-NEXT:    switch_4 ? _GEN_1[switch_3] : switch_2 ? _GEN_0[switch_1] : _GEN[switch_0];
// SV-NEXT:endmodule

// CHECK: builtin.module {
// CHECK-NEXT: hw.module private @myfirstaccelerator(in %data data_0: i32, in %data_1: i32, in %switch switch_0: i2, in %switch_1: i1, in %switch_2: i1, in %switch_3: i1, in %switch_4: i1, in %switch_5: i1, out out_0: i32
// CHECK-NEXT:     %0 = comb.mul %data, %data_1 : i32
// CHECK-NEXT:     %1 = comb.add %data, %data_1 : i32
// CHECK-NEXT:     %2 = comb.sub %data, %data_1 : i32
// CHECK-NEXT:     %3 = hw.array_create %0, %1, %2 : i32
// CHECK-NEXT:     %4 = hw.array_get %3[%switch] : !hw.array<3xi32>, i2
// CHECK-NEXT:     %5 = comb.mul %data, %4 : i32
// CHECK-NEXT:     %6 = comb.add %data, %4 : i32
// CHECK-NEXT:     %7 = hw.array_create %5, %6 : i32
// CHECK-NEXT:     %8 = hw.array_get %7[%switch_1] : !hw.array<2xi32>, i1
// CHECK-NEXT:     %9 = comb.mux %switch_2, %8, %4 : i32
// CHECK-NEXT:     %10 = comb.mux %switch_5, %data_1, %8 : i32
// CHECK-NEXT:     %11 = comb.mul %4, %8 : i32
// CHECK-NEXT:     %12 = comb.divu %4, %10 : i32
// CHECK-NEXT:     %13 = hw.array_create %11, %12 : i32
// CHECK-NEXT:     %14 = hw.array_get %13[%switch_3] : !hw.array<2xi32>, i1
// CHECK-NEXT:     %15 = comb.mux %switch_4, %14, %9 : i32
// CHECK-NEXT:     hw.output %15 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }
