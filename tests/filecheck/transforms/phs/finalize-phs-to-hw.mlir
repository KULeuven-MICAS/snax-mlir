// RUN: snax-opt %s -p finalize-phs-to-hw | circt-opt --map-arith-to-comb | firtool --format=mlir --strip-debug-info | filecheck %s --check-prefix=SV
// RUN: snax-opt %s -p finalize-phs-to-hw | circt-opt --map-arith-to-comb | filecheck %s

builtin.module {
  hw.module private @myfirstaccelerator(in %0 data_0: i32, in %1 data_1: i32, in %2 switch_0: i2, in %3 switch_1: i1, in %4 switch_2: i1, in %5 switch_3: i1, in %6 switch_4: i1, in %7 switch_5: i1, out out_0: i32) {
    %8 = builtin.unrealized_conversion_cast %2 : i2 to index
    %9 = builtin.unrealized_conversion_cast %3 : i1 to index
    %10 = builtin.unrealized_conversion_cast %4 : i1 to index
    %11 = builtin.unrealized_conversion_cast %5 : i1 to index
    %12 = builtin.unrealized_conversion_cast %6 : i1 to index
    %13 = builtin.unrealized_conversion_cast %7 : i1 to index
    %14 = phs.choose @_0 with %8 (%0 : i32, %1 : i32) -> i32
      0) (%15, %16) {
        %17 = arith.muli %15, %16 : i32
        phs.yield %17 : i32
      }
      1) (%18, %19) {
        %20 = arith.addi %18, %19 : i32
        phs.yield %20 : i32
      }
      2) (%21, %22) {
        %23 = arith.subi %21, %22 : i32
        phs.yield %23 : i32
      }
    %15 = phs.choose @_1 with %9 (%0 : i32, %14 : i32) -> i32
      0) (%16, %17) {
        %18 = arith.muli %16, %17 : i32
        phs.yield %18 : i32
      }
      1) (%19, %20) {
        %21 = arith.addi %19, %20 : i32
        phs.yield %21 : i32
      }
    %16 = phs.mux with %10 (%14 : i32, %15 : i32) -> i32
    %17 = phs.mux with %13 (%15 : i32, %1 : i32) -> i32
    %18 = phs.choose @_2 with %11 (%14 : i32, %17 : i32) -> i32
      0) (%19, %20) {
        %21 = arith.muli %19, %20 : i32
        phs.yield %21 : i32
      }
      1) (%22, %23) {
        %24 = arith.divui %22, %23 : i32
        phs.yield %24 : i32
      }
    %19 = phs.mux with %12 (%16 : i32, %18 : i32) -> i32
    hw.output %19 : i32
  }
}

// SV: module myfirstaccelerator(
// SV-NEXT:   input  [31:0] data_0,
// SV-NEXT:                 data_1,
// SV-NEXT:   input  [1:0]  switch_0,
// SV-NEXT:   input         switch_1,
// SV-NEXT:                 switch_2,
// SV-NEXT:                 switch_3,
// SV-NEXT:                 switch_4,
// SV-NEXT:                 switch_5,
// SV-NEXT:   output [31:0] out_0
// SV-NEXT: );

// SV:   wire [2:0][31:0] _GEN = {{[{][{]}}data_0 - data_1}, {data_0 + data_1}, {data_0 * data_1{{[}][}]}};
// SV-NEXT:   wire [1:0][31:0] _GEN_0 = {{[{][{]}}data_0 + _GEN[switch_0]}, {data_0 * _GEN[switch_0]{{[}][}]}};
// SV-NEXT:   wire [31:0]      _GEN_1 = switch_5 ? data_1 : _GEN_0[switch_1];
// SV-NEXT:   wire [1:0][31:0] _GEN_2 = {{[{][{]}}_GEN[switch_0] / _GEN_1}, {_GEN[switch_0] * _GEN_1{{[}][}]}};
// SV-NEXT:   assign out_0 =
// SV-NEXT:     switch_4 ? _GEN_2[switch_3] : switch_2 ? _GEN_0[switch_1] : _GEN[switch_0];
// SV-NEXT: endmodule


// CHECK: module {
// CHECK-NEXT:   hw.module private @myfirstaccelerator(in %data_0 : i32, in %data_1 : i32, in %switch_0 : i2, in %switch_1 : i1, in %switch_2 : i1, in %switch_3 : i1, in %switch_4 : i1, in %switch_5 : i1, out out_0 : i32) {
// CHECK-NEXT:     %0 = comb.mul %data_0, %data_1 : i32
// CHECK-NEXT:     %1 = comb.add %data_0, %data_1 : i32
// CHECK-NEXT:     %2 = comb.sub %data_0, %data_1 : i32
// CHECK-NEXT:     %3 = hw.array_create %2, %1, %0 : i32
// CHECK-NEXT:     %4 = hw.array_get %3[%switch_0] : !hw.array<3xi32>, i2
// CHECK-NEXT:     %5 = comb.mul %data_0, %4 : i32
// CHECK-NEXT:     %6 = comb.add %data_0, %4 : i32
// CHECK-NEXT:     %7 = hw.array_create %6, %5 : i32
// CHECK-NEXT:     %8 = hw.array_get %7[%switch_1] : !hw.array<2xi32>, i1
// CHECK-NEXT:     %9 = comb.mux %switch_2, %8, %4 : i32
// CHECK-NEXT:     %10 = comb.mux %switch_5, %data_1, %8 : i32
// CHECK-NEXT:     %11 = comb.mul %4, %10 : i32
// CHECK-NEXT:     %12 = comb.divu %4, %10 : i32
// CHECK-NEXT:     %13 = hw.array_create %12, %11 : i32
// CHECK-NEXT:     %14 = hw.array_get %13[%switch_3] : !hw.array<2xi32>, i1
// CHECK-NEXT:     %15 = comb.mux %switch_4, %14, %9 : i32
// CHECK-NEXT:     hw.output %15 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }
