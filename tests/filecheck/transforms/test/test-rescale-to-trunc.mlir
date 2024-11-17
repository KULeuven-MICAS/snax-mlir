// RUN: snax-opt --split-input-file -p test-rescale-to-trunc %s | filecheck %s

%in = "test.op" () : () -> i32
%a = "kernel.rescale"(%in) {double_round = true, input_zp = 23 : i8, max_int = 100 : i8, min_int = -110 : i8, multiplier = 1234567890 : i32, output_zp = -15 : i8, shift = 39 : i8} : (i32) -> i8

// CHECK:   %a = arith.trunci %in : i32 to i8
