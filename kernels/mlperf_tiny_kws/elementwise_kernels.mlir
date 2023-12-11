// bias
%4 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_21, %3 : tensor<64xi32>, tensor<?x25x5x64xi32>) outs(%2 : tensor<?x25x5x64xi32>) {
^bb0(%in: i32, %in_179: i32, %out: i32):
    %132 = arith.addi %in, %in_179 : i32
    linalg.yield %132 : i32
} -> tensor<?x25x5x64xi32>

// scale + clamp
 %6 = linalg.generic {indexing_maps = [#map1, #map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %cst_26, %cst_27 : tensor<?x25x5x64xi32>, tensor<64xi32>, tensor<64xi8>) outs(%5 : tensor<?x25x5x64xi8>) {
^bb0(%in: i32, %in_179: i32, %in_180: i8, %out: i8):
    %c0_i32_181 = arith.constant 0 : i32
    %c-128_i32_182 = arith.constant -128 : i32
    %132 = arith.subi %in, %c0_i32_181 : i32
    %133 = tosa.apply_scale %132, %in_179, %in_180 {double_round = true} : (i32, i32, i8) -> i32
    %134 = arith.addi %133, %c-128_i32_182 : i32
    %c-128_i32_183 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %135 = arith.cmpi slt, %134, %c-128_i32_183 : i32
    %136 = arith.select %135, %c-128_i32_183, %134 : i32
    %137 = arith.cmpi slt, %c127_i32, %134 : i32
    %138 = arith.select %137, %c127_i32, %136 : i32
    %139 = arith.trunci %138 : i32 to i8
    linalg.yield %139 : i8
} -> tensor<?x25x5x64xi8>

// elaborate nop
%8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<?x25x5x64xi8>) outs(%7 : tensor<?x25x5x64xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c-128_i8_179 = arith.constant -128 : i8
      %c127_i8 = arith.constant 127 : i8
      %132 = arith.cmpi slt, %in, %c-128_i8_179 : i8
      %133 = arith.select %132, %c-128_i8_179, %in : i8
      %134 = arith.cmpi slt, %c127_i8, %in : i8
      %135 = arith.select %134, %c127_i8, %133 : i8
      linalg.yield %135 : i8
    } -> tensor<?x25x5x64xi8>
