// RUN: snax-opt -p convert-tosa-to-kernel %s | filecheck %s

%0 = "test.op"() : () -> tensor<?x8xi32>
%1 = tosa.rescale %0 {"double_round" = true, "input_zp" = 0 : i32, "multiplier" = array<i32: 1085889731>, "output_zp" = -128 : i32, "per_channel" = false, "scale32" = true, "shift" = array<i32: 37>} : (tensor<?x8xi32>) -> tensor<?x8xi8>
%2 = tosa.clamp %1 {"max_fp" = 0.000000e+00 : f32, "max_int" = 127 : i64, "min_fp" = 0.000000e+00 : f32, "min_int" = -128 : i64} : (tensor<?x8xi8>) -> tensor<?x8xi8>

// CHECK: builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> tensor<?x8xi32>
// CHECK-NEXT:   %1 = arith.constant 0 : index
// CHECK-NEXT:   %2 = tensor.dim %0, %1 : tensor<?x8xi32>
// CHECK-NEXT:   %3 = tensor.empty(%2) : tensor<?x8xi8>
// CHECK-NEXT:   %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<?x8xi32>) outs(%3 : tensor<?x8xi8>) {
// CHECK-NEXT:   ^bb0(%5 : i32, %6 : i8):
// CHECK-NEXT:     %7 = kernel.rescale %5 {input_zp = 0 : i32, output_zp = -128 : i32, multiplier = array<i32: 1085889731>, shift = array<i32: 37>, max_int = 127 : i32, min_int = -128 : i32, double_round = true} : (i32) -> i8
// CHECK-NEXT:     linalg.yield %7 : i8
// CHECK-NEXT:   } -> tensor<?x8xi8>
// CHECK-NEXT: }
