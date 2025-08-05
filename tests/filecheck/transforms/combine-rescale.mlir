// RUN: snax-opt %s -p combine-rescale --split-input-file | filecheck %s

func.func public @rescale_combined(%arg0: tensor<64xi8>) -> tensor<64xi32> {
%0 = tosa.rescale %arg0 {double_round = true, input_zp = 38 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 10>} : (tensor<64xi8>) -> tensor<64xi32>
%1 = tosa.rescale %0 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1657902019>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 33>} : (tensor<64xi32>) -> tensor<64xi32>
return %1 : tensor<64xi32>
}

// CHECK:  func.func public @rescale_combined(%arg0 : tensor<64xi8>) -> tensor<64xi32> {
// CHECK-NEXT:    %0 = tosa.rescale %arg0 {input_zp = 38 : i32, output_zp = 0 : i32, multiplier = array<i32: 1657902019>, shift = array<i8: 13>, scale32 = true, double_round = true, per_channel = false} : (tensor<64xi8>) -> tensor<64xi32>
// CHECK-NEXT:    func.return %0 : tensor<64xi32>
// CHECK-NEXT:  }


// -----
func.func public @rescale_combined(%arg0: tensor<64xi8>) -> tensor<64xi32> {
%0 = tosa.rescale %arg0 {double_round = true, input_zp = 38 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 10>} : (tensor<64xi8>) -> tensor<64xi32>
%1 = tosa.rescale %0 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 16>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 4>} : (tensor<64xi32>) -> tensor<64xi32>
%2 = tosa.rescale %1 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1657902019>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 33>} : (tensor<64xi32>) -> tensor<64xi32>
return %2 : tensor<64xi32>
}


// CHECK:  func.func public @rescale_combined(%arg0 : tensor<64xi8>) -> tensor<64xi32> {
// CHECK-NEXT:    %0 = tosa.rescale %arg0 {input_zp = 38 : i32, output_zp = 0 : i32, multiplier = array<i32: 1657902019>, shift = array<i8: 13>, scale32 = true, double_round = true, per_channel = false} : (tensor<64xi8>) -> tensor<64xi32>
// CHECK-NEXT:    func.return %0 : tensor<64xi32>
// CHECK-NEXT:  }

// -----
func.func public @rescale_combined(%arg0: i8) -> i32 {
  %0 = "kernel.rescale"(%arg0) {input_zp = 38 : i32, output_zp = 0 : i32, multiplier = array<i32: 1073741824>, shift = array<i32: 10>, min_int = -2147483648 : i32, max_int = 2147483647 : i32, double_round = true} : (i8) -> i32
  %1 = "kernel.rescale"(%0) {input_zp = 0 : i32, output_zp = 0 : i32, multiplier = array<i32: 1657902019>, shift = array<i32: 33>, min_int = -2147483648 : i32, max_int = 2147483647 : i32, double_round = true} : (i32) -> i32
  return %1 : i32
}


// CHECK:  func.func public @rescale_combined(%arg0 : i8) -> i32 {
// CHECK-NEXT:    %0 = kernel.rescale %arg0 {input_zp = 38 : i32, output_zp = 0 : i32, multiplier = array<i32: 1657902019>, shift = array<i32: 13>, max_int = 2147483647 : i32, min_int = -2147483648 : i32, double_round = true} : (i8) -> i32
// CHECK-NEXT:    func.return %0 : i32
// CHECK-NEXT:  }


// -----
func.func public @rescale_combined(%arg0: i8) -> i32 {
  %0 = "kernel.rescale"(%arg0) {input_zp = 38 : i32, output_zp = 0 : i32, multiplier = array<i32: 1073741824>, shift = array<i32: 10>, min_int = -2147483648 : i32, max_int = 2147483647 : i32, double_round = true} : (i8) -> i32
  %1 = "kernel.rescale"(%0) {input_zp = 0 : i32, output_zp = 0 : i32, multiplier = array<i32: 16>, shift = array<i32: 4>, min_int = -2147483648 : i32, max_int = 2147483647 : i32, double_round = true} : (i32) -> i32
  %2 = "kernel.rescale"(%1) {input_zp = 0 : i32, output_zp = 0 : i32, multiplier = array<i32: 1657902019>, shift = array<i32: 33>, min_int = -2147483648 : i32, max_int = 2147483647 : i32, double_round = true} : (i32) -> i32
  return %2 : i32
}

// CHECK:  func.func public @rescale_combined(%arg0 : i8) -> i32 {
// CHECK-NEXT:    %0 = kernel.rescale %arg0 {input_zp = 38 : i32, output_zp = 0 : i32, multiplier = array<i32: 1657902019>, shift = array<i32: 13>, max_int = 2147483647 : i32, min_int = -2147483648 : i32, double_round = true} : (i8) -> i32
// CHECK-NEXT:    func.return %0 : i32
// CHECK-NEXT:  }




