// RUN: snax-opt %s -p phs-export-to-verilog{output=\"output.sv\"} | filecheck %s --check-prefix=STDOUT
// RUN: filecheck %s --check-prefix=FILE --input-file=output.sv

builtin.module {
  hw.module @acc1(in %arg0 data_0: i64, in %arg1 data_1: i64, in %arg2 data_2: i64, in %arg3 switch_0: i0, out out_0: i64) {
    %0 = comb.add %arg0, %arg1 : i64
    %1 = hw.array_create %0 : i64
    %2 = hw.array_get %1[%arg3] : !hw.array<1xi64>, i0
    hw.output %2 : i64
  }
}

// STDOUT: builtin.module {
// STDOUT-NEXT:   hw.module @acc1(in %arg0 data_0: i64, in %arg1 data_1: i64, in %arg2 data_2: i64, in %arg3 switch_0: i0, out out_0: i64) {
// STDOUT-NEXT:     %0 = comb.add %arg0, %arg1 : i64
// STDOUT-NEXT:     %1 = hw.array_create %0 : i64
// STDOUT-NEXT:     %2 = hw.array_get %1[%arg3] : !hw.array<1xi64>, i0
// STDOUT-NEXT:     hw.output %2 : i64
// STDOUT-NEXT:   }
// STDOUT-NEXT: }

// FILE: module acc1(	// <stdin>:2:3
// FILE-NEXT:      input  [63:0]         data_0,	// <stdin>:3:8
// FILE-NEXT:                            data_1,	// <stdin>:3:21
// FILE-NEXT:                            data_2,	// <stdin>:3:34
// FILE-NEXT:   // input  /*Zero Width*/ switch_0,	// <stdin>:3:47
// FILE-NEXT:      output [63:0]         out_0
// FILE-NEXT: );
// FILE:   assign out_0 = data_0 + data_1;	// <stdin>:4:10, :7:5
// FILE-NEXT: endmodule
