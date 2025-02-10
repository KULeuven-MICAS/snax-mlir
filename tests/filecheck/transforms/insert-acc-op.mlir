// RUN: snax-opt  %s -p insert-accfg-op{accelerator=snax_hwpe_mult} | filecheck %s

builtin.module{}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   "accfg.accelerator"() <{name = @snax_hwpe_mult, fields = {A = 976 : i32, B = 977 : i32, O = 979 : i32, vector_length = 980 : i32, nr_iters = 981 : i32, mode = 982 : i32}, launch_fields = {launch = 960 : i32}, barrier = 963 : i32}> : () -> ()
// CHECK-NEXT: }


