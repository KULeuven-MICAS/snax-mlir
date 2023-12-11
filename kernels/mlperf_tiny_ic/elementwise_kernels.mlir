// bias
%4 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_21, %3 : tensor<16xi32>, tensor<?x32x32x16xi32>) outs(%2 : tensor<?x32x32x16xi32>) {
^bb0(%in: i32, %in_231: i32, %out: i32):
    %173 = arith.addi %in, %in_231 : i32
    linalg.yield %173 : i32
} -> tensor<?x32x32x16xi32>

// scale + clamp
%6 = linalg.generic {indexing_maps = [#map1, #map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %cst_26, %cst_27 : tensor<?x32x32x16xi32>, tensor<16xi32>, tensor<16xi8>) outs(%5 : tensor<?x32x32x16xi8>) {
^bb0(%in: i32, %in_231: i32, %in_232: i8, %out: i8):
    %c0_i32_233 = arith.constant 0 : i32
    %c-128_i32_234 = arith.constant -128 : i32
    %173 = arith.subi %in, %c0_i32_233 : i32
    %174 = tosa.apply_scale %173, %in_231, %in_232 {double_round = true} : (i32, i32, i8) -> i32
    %175 = arith.addi %174, %c-128_i32_234 : i32
    %c-128_i32_235 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %176 = arith.cmpi slt, %175, %c-128_i32_235 : i32
    %177 = arith.select %176, %c-128_i32_235, %175 : i32
    %178 = arith.cmpi slt, %c127_i32, %175 : i32
    %179 = arith.select %178, %c127_i32, %177 : i32
    %180 = arith.trunci %179 : i32 to i8
    linalg.yield %180 : i8
} -> tensor<?x32x32x16xi8>

// the elaborate do nothing operation
%8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<?x32x32x16xi8>) outs(%7 : tensor<?x32x32x16xi8>) {
^bb0(%in: i8, %out: i8):
    %c-128_i8_231 = arith.constant -128 : i8
    %c127_i8 = arith.constant 127 : i8
    %173 = arith.cmpi slt, %in, %c-128_i8_231 : i8
    %174 = arith.select %173, %c-128_i8_231, %in : i8
    %175 = arith.cmpi slt, %c127_i8, %in : i8
    %176 = arith.select %175, %c127_i8, %174 : i8
    linalg.yield %176 : i8
} -> tensor<?x32x32x16xi8>


// average pooling operation 2d
#map = affine_map<(d0, d1, d2, d3) -> ()>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 8 + d4, d2 * 8 + d5, d3)>
#map10 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
#map11 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>

%125 = linalg.generic {indexing_maps = [#map9, #map10, #map11], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%121, %124 : tensor<?x8x8x64xi8>, tensor<8x8xi32>) outs(%123 : tensor<?x1x1x64xi32>) {
^bb0(%in: i8, %in_123: i32, %out: i32):
    %173 = arith.extsi %in : i8 to i32
    %174 = arith.addi %out, %173 : i32
    linalg.yield %174 : i32
} -> tensor<?x1x1x64xi32>
%127 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%125 : tensor<?x1x1x64xi32>) outs(%126 : tensor<?x1x1x64xi8>) {
^bb0(%in: i32, %out: i8):
    %173 = arith.extsi %in : i32 to i64
    %174 = arith.muli %173, %c1073741825_i64 : i64
    %175 = arith.addi %174, %c34359738368_i64 : i64
    %176 = arith.shrsi %175, %c36_i64 : i64
    %177 = arith.trunci %176 : i64 to i32
    %178 = arith.cmpi slt, %177, %c-128_i32 : i32
    %179 = arith.select %178, %c-128_i32, %177 : i32
    %180 = arith.cmpi sgt, %177, %c127_i32 : i32
    %181 = arith.select %180, %c127_i32, %179 : i32
    %182 = arith.trunci %181 : i32 to i8
    linalg.yield %182 : i8
} -> tensor<?x1x1x64xi8>

