// RUN: snax-opt -p convert-tosa-to-kernel %s | filecheck %s

%0 = "test.op"() : () -> tensor<?x8xi32>
%input_zp = "tosa.const"() <{ values = dense<0> : tensor<1xi32> }> : () -> tensor<1xi32>
%output_zp = "tosa.const"() <{ values = dense<-128> : tensor<1xi32> }> : () -> tensor<1xi32>
%multiplier = "tosa.const"() <{ values = dense<1085889731> : tensor<1xi32> }> : () -> tensor<1xi32>
%shift = "tosa.const"() <{ values = dense<37> : tensor<1xi32> }> : () -> tensor<1xi32>
%1 = tosa.rescale %0, %multiplier, %shift, %input_zp, %output_zp {"rounding_mode" = "DOUBLE_ROUND", "per_channel" = false, "scale32" = true, "input_unsigned" = false, output_unsigned = false} : (tensor<?x8xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?x8xi8>
%2 = tosa.clamp %1 {"max_val" = 127 : i8, "min_val" = -128 : i8} : (tensor<?x8xi8>) -> tensor<?x8xi8>

// CHECK:        %{{.*}} = tensor.empty() : tensor<?x8xi8>
// CHECK-NEXT:   %{{.*}} = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%{{.*}} : tensor<?x8xi32>) outs(%{{.*}} : tensor<?x8xi8>) {
// CHECK-NEXT:   ^bb0(%3 : i32, %4 : i8):
// CHECK-NEXT:     %{{.*}} = kernel.rescale %{{.*}} {input_zp = 0 : i32, output_zp = -128 : i32, multiplier = array<i32: 1085889731>, shift = array<i32: 37>, max_int = 127 : i32, min_int = -128 : i32, double_round = true} : (i32) -> i8
// CHECK-NEXT:     linalg.yield %{{.*}} : i8
// CHECK-NEXT:   } -> tensor<?x8xi8>
