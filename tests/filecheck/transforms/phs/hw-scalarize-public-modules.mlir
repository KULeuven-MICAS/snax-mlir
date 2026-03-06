// RUN: snax-opt %s -p hw-scalarize-public-modules | circt-opt --allow-unregistered-dialect --canonicalize | filecheck %s

hw.module @test_1d(in %data data_0: !hw.array<4xi64>, in %data_1: !hw.array<4xi64>, in %switch switch_0: i2, out out_0: !hw.array<4xi64>) {
  %0 = "test.op"(%data, %data_1, %switch) : (!hw.array<4xi64>, !hw.array<4xi64>, i2) -> !hw.array<4xi64>
  hw.output %0 : !hw.array<4xi64>
}

hw.module @test_2d(in %data data_0: !hw.array<2x!hw.array<4xi64>>, in %data_1: !hw.array<2x!hw.array<4xi64>>, out out_0: !hw.array<2x!hw.array<4xi64>>) {
  hw.output %data : !hw.array<2x!hw.array<4xi64>>
}

// CHECK: hw.module @test_1d(in %data_0_0 : i64, in %data_0_1 : i64, in %data_0_2 : i64, in %data_0_3 : i64, in %data_1_0 : i64, in %data_1_1 : i64, in %data_1_2 : i64, in %data_1_3 : i64, in %switch_0 : i2, out out_0_0 : i64, out out_0_1 : i64, out out_0_2 : i64, out out_0_3 : i64) {
// CHECK:   hw.output %3, %4, %5, %6 : i64, i64, i64, i64
// CHECK-NEXT: }
// CHECK: hw.module @test_2d(in %data_0_0 : i64, in %data_0_1 : i64, in %data_0_2 : i64, in %data_0_3 : i64, in %data_0_4 : i64, in %data_0_5 : i64, in %data_0_6 : i64, in %data_0_7 : i64, in %data_1_0 : i64, in %data_1_1 : i64, in %data_1_2 : i64, in %data_1_3 : i64, in %data_1_4 : i64, in %data_1_5 : i64, in %data_1_6 : i64, in %data_1_7 : i64, out out_0_0_0 : i64, out out_0_0_1 : i64, out out_0_0_2 : i64, out out_0_0_3 : i64, out out_0_1_0 : i64, out out_0_1_1 : i64, out out_0_1_2 : i64, out out_0_1_3 : i64) {
// CHECK-NEXT:   hw.output %data_0_0, %data_0_1, %data_0_2, %data_0_3, %data_0_4, %data_0_5, %data_0_6, %data_0_7 : i64, i64, i64, i64, i64, i64, i64, i64
// CHECK-NEXT: }
