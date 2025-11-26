// RUN: snax-opt %s -p tosa-combine-rescale --split-input-file | filecheck %s

func.func public @rescale_combined(%arg0: tensor<64xi8>) -> tensor<64xi32> {
    %input_zp = "tosa.const"() <{ values = dense<38> : tensor<1xi32> }> : () -> tensor<1xi32>
    %output_zp = "tosa.const"() <{ values = dense<0> : tensor<1xi32> }> : () -> tensor<1xi32>
    %multiplier = "tosa.const"() <{ values = dense<1073741824> : tensor<1xi32> }> : () -> tensor<1xi32>
    %shift = "tosa.const"() <{ values = dense<10> : tensor<1xi32> }> : () -> tensor<1xi32>
    %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {"rounding_mode" = "DOUBLE_ROUND", "per_channel" = false, "scale32" = true, "input_unsigned" = false, output_unsigned = false} : (tensor<64xi8>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<64xi32>
    %input_zp_2 = "tosa.const"() <{ values = dense<0> : tensor<1xi32> }> : () -> tensor<1xi32>
    %output_zp_2 = "tosa.const"() <{ values = dense<0> : tensor<1xi32> }> : () -> tensor<1xi32>
    %multiplier_2 = "tosa.const"() <{ values = dense<1657902019> : tensor<1xi32> }> : () -> tensor<1xi32>
    %shift_2 = "tosa.const"() <{ values = dense<33> : tensor<1xi32> }> : () -> tensor<1xi32>
    %1 = tosa.rescale %0, %multiplier_2, %shift_2, %input_zp_2, %output_zp_2 {"rounding_mode" = "DOUBLE_ROUND", "per_channel" = false, "scale32" = true, "input_unsigned" = false, output_unsigned = false} : (tensor<64xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<64xi32>
    return %1 : tensor<64xi32>
}

// CHECK:         %0 = "tosa.const"() <{values = dense<38> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK-NEXT:    %1 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK-NEXT:    %2 = "tosa.const"() <{values = dense<1657902019> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK-NEXT:    %3 = "tosa.const"() <{values = dense<13> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK-NEXT:    %4 = tosa.rescale %arg0, %2, %3, %0, %1 {rounding_mode = "DOUBLE_ROUND", per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<64xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<64xi32>
// CHECK-NEXT:    func.return %4 : tensor<64xi32>
