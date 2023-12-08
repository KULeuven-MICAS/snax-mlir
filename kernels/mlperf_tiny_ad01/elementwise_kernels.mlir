// fill
%3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_18 : tensor<128x640xi8>) outs(%2 : tensor<640x128xi8>) {
^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
} -> tensor<640x128xi8>

// bias before quantized matmul
%6 = linalg.generic {indexing_maps = [#map2, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_17, %5 : tensor<128xi32>, tensor<?x128xi32>) outs(%4 : tensor<?x128xi32>) {
^bb0(%in: i32, %in_116: i32, %out: i32):
    %108 = arith.addi %in, %in_116 : i32
    linalg.yield %108 : i32
} -> tensor<?x128xi32>

// rescale + clamp?
%8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<?x128xi32>) outs(%7 : tensor<?x128xi8>) {
^bb0(%in: i32, %out: i8):
    %c0_i32_116 = arith.constant 0 : i32
    %c-128_i32_117 = arith.constant -128 : i32
    // zero point adjustment
    %108 = arith.subi %in, %c0_i32_116 : i32

    // scaling
    %109 = tosa.apply_scale %108, %c1638001719_i32, %c39_i8 {double_round = true} : (i32, i32, i8) -> i32

    // zero point adjustment
    %110 = arith.addi %109, %c-128_i32_117 : i32
    %c-128_i32_118 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32

    // clamping lower boundary
    %111 = arith.cmpi slt, %110, %c-128_i32_118 : i32
    %112 = arith.select %111, %c-128_i32_118, %110 : i32

    // clamping upper boundary
    %113 = arith.cmpi slt, %c127_i32, %110 : i32
    %114 = arith.select %113, %c127_i32, %112 : i32

    // signed truncation
    %115 = arith.trunci %114 : i32 to i8
    linalg.yield %115 : i8
} -> tensor<?x128xi8>

// like literally what the fuck this doesn't do anything
// this clamps an i8 type between -128 and 127? what else would the value be? 
%10 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<?x128xi8>) outs(%9 : tensor<?x128xi8>) {
^bb0(%in: i8, %out: i8):
    %c-128_i8 = arith.constant -128 : i8
    %c127_i8 = arith.constant 127 : i8
    %108 = arith.cmpi slt, %in, %c-128_i8 : i8
    %109 = arith.select %108, %c-128_i8, %in : i8
    %110 = arith.cmpi slt, %c127_i8, %in : i8
    %111 = arith.select %110, %c127_i8, %109 : i8
    linalg.yield %111 : i8
} -> tensor<?x128xi8>

// fill
%14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_16 : tensor<128x128xi8>) outs(%13 : tensor<128x128xi8>) {
^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
} -> tensor<128x128xi8>

// bias
%17 = linalg.generic {indexing_maps = [#map2, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_15, %16 : tensor<128xi32>, tensor<?x128xi32>) outs(%15 : tensor<?x128xi32>) {
^bb0(%in: i32, %in_116: i32, %out: i32):
    %108 = arith.addi %in, %in_116 : i32
    linalg.yield %108 : i32
} -> tensor<?x128xi32>

// scale + clamp
%19 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%17 : tensor<?x128xi32>) outs(%18 : tensor<?x128xi8>) {
^bb0(%in: i32, %out: i8):
    %c0_i32_116 = arith.constant 0 : i32
    %c-128_i32_117 = arith.constant -128 : i32
    %108 = arith.subi %in, %c0_i32_116 : i32
    %109 = tosa.apply_scale %108, %c1442659867_i32, %c36_i8 {double_round = true} : (i32, i32, i8) -> i32
    %110 = arith.addi %109, %c-128_i32_117 : i32
    %c-128_i32_118 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %111 = arith.cmpi slt, %110, %c-128_i32_118 : i32
    %112 = arith.select %111, %c-128_i32_118, %110 : i32
    %113 = arith.cmpi slt, %c127_i32, %110 : i32
    %114 = arith.select %113, %c127_i32, %112 : i32
    %115 = arith.trunci %114 : i32 to i8
    linalg.yield %115 : i8
} -> tensor<?x128xi8>
%21 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%19 : tensor<?x128xi8>) outs(%20 : tensor<?x128xi8>) {
^bb0(%in: i8, %out: i8):
    %c-128_i8 = arith.constant -128 : i8
    %c127_i8 = arith.constant 127 : i8
    %108 = arith.cmpi slt, %in, %c-128_i8 : i8
    %109 = arith.select %108, %c-128_i8, %in : i8
    %110 = arith.cmpi slt, %c127_i8, %in : i8
    %111 = arith.select %110, %c127_i8, %109 : i8
    linalg.yield %111 : i8
} -> tensor<?x128xi8>
%25 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_14 : tensor<128x128xi8>) outs(%24 : tensor<128x128xi8>) {
^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
} -> tensor<128x128xi8>
%28 = linalg.generic {indexing_maps = [#map2, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_13, %27 : tensor<128xi32>, tensor<?x128xi32>) outs(%26 : tensor<?x128xi32>) {
^bb0(%in: i32, %in_116: i32, %out: i32):
    %108 = arith.addi %in, %in_116 : i32
    linalg.yield %108 : i32
} -> tensor<?x128xi32>
%30 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%28 : tensor<?x128xi32>) outs(%29 : tensor<?x128xi8>) {
^bb0(%in: i32, %out: i8):
    %c0_i32_116 = arith.constant 0 : i32
    %c-128_i32_117 = arith.constant -128 : i32
    %108 = arith.subi %in, %c0_i32_116 : i32
    %109 = tosa.apply_scale %108, %c1185020333_i32, %c33_i8 {double_round = true} : (i32, i32, i8) -> i32
    %110 = arith.addi %109, %c-128_i32_117 : i32
    %c-128_i32_118 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %111 = arith.cmpi slt, %110, %c-128_i32_118 : i32
    %112 = arith.select %111, %c-128_i32_118, %110 : i32
    %113 = arith.cmpi slt, %c127_i32, %110 : i32
    %114 = arith.select %113, %c127_i32, %112 : i32
    %115 = arith.trunci %114 : i32 to i8
    linalg.yield %115 : i8
} -> tensor<?x128xi8>
%32 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%30 : tensor<?x128xi8>) outs(%31 : tensor<?x128xi8>) {
^bb0(%in: i8, %out: i8):
    %c-128_i8 = arith.constant -128 : i8
    %c127_i8 = arith.constant 127 : i8
    %108 = arith.cmpi slt, %in, %c-128_i8 : i8
    %109 = arith.select %108, %c-128_i8, %in : i8
    %110 = arith.cmpi slt, %c127_i8, %in : i8
    %111 = arith.select %110, %c127_i8, %109 : i8
    linalg.yield %111 : i8
} -> tensor<?x128xi8>
%36 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_12 : tensor<128x128xi8>) outs(%35 : tensor<128x128xi8>) {
^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
} -> tensor<128x128xi8>
%39 = linalg.generic {indexing_maps = [#map2, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_11, %38 : tensor<128xi32>, tensor<?x128xi32>) outs(%37 : tensor<?x128xi32>) {
^bb0(%in: i32, %in_116: i32, %out: i32):
    %108 = arith.addi %in, %in_116 : i32
    linalg.yield %108 : i32
} -> tensor<?x128xi32>
%41 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%39 : tensor<?x128xi32>) outs(%40 : tensor<?x128xi8>) {
^bb0(%in: i32, %out: i8):
    %c0_i32_116 = arith.constant 0 : i32
    %c-128_i32_117 = arith.constant -128 : i32
    %108 = arith.subi %in, %c0_i32_116 : i32
    %109 = tosa.apply_scale %108, %c1439819856_i32, %c35_i8 {double_round = true} : (i32, i32, i8) -> i32
    %110 = arith.addi %109, %c-128_i32_117 : i32
    %c-128_i32_118 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %111 = arith.cmpi slt, %110, %c-128_i32_118 : i32
    %112 = arith.select %111, %c-128_i32_118, %110 : i32
    %113 = arith.cmpi slt, %c127_i32, %110 : i32
    %114 = arith.select %113, %c127_i32, %112 : i32
    %115 = arith.trunci %114 : i32 to i8
    linalg.yield %115 : i8
} -> tensor<?x128xi8>
%43 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%41 : tensor<?x128xi8>) outs(%42 : tensor<?x128xi8>) {
^bb0(%in: i8, %out: i8):
    %c-128_i8 = arith.constant -128 : i8
    %c127_i8 = arith.constant 127 : i8
    %108 = arith.cmpi slt, %in, %c-128_i8 : i8
    %109 = arith.select %108, %c-128_i8, %in : i8
    %110 = arith.cmpi slt, %c127_i8, %in : i8
    %111 = arith.select %110, %c127_i8, %109 : i8
    linalg.yield %111 : i8
} -> tensor<?x128xi8>
%47 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_10 : tensor<8x128xi8>) outs(%46 : tensor<128x8xi8>) {
^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
} -> tensor<128x8xi8>
%50 = linalg.generic {indexing_maps = [#map2, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_9, %49 : tensor<8xi32>, tensor<?x8xi32>) outs(%48 : tensor<?x8xi32>) {
^bb0(%in: i32, %in_116: i32, %out: i32):
    %108 = arith.addi %in, %in_116 : i32
    linalg.yield %108 : i32
} -> tensor<?x8xi32>
%52 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%50 : tensor<?x8xi32>) outs(%51 : tensor<?x8xi8>) {
^bb0(%in: i32, %out: i8):
    %c0_i32_116 = arith.constant 0 : i32
    %c-128_i32_117 = arith.constant -128 : i32
    %108 = arith.subi %in, %c0_i32_116 : i32
    %109 = tosa.apply_scale %108, %c1085889731_i32, %c37_i8 {double_round = true} : (i32, i32, i8) -> i32
    %110 = arith.addi %109, %c-128_i32_117 : i32
    %c-128_i32_118 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %111 = arith.cmpi slt, %110, %c-128_i32_118 : i32
    %112 = arith.select %111, %c-128_i32_118, %110 : i32
    %113 = arith.cmpi slt, %c127_i32, %110 : i32
    %114 = arith.select %113, %c127_i32, %112 : i32
    %115 = arith.trunci %114 : i32 to i8
    linalg.yield %115 : i8
} -> tensor<?x8xi8>
%54 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%52 : tensor<?x8xi8>) outs(%53 : tensor<?x8xi8>) {
^bb0(%in: i8, %out: i8):
    %c-128_i8 = arith.constant -128 : i8
    %c127_i8 = arith.constant 127 : i8
    %108 = arith.cmpi slt, %in, %c-128_i8 : i8
    %109 = arith.select %108, %c-128_i8, %in : i8
    %110 = arith.cmpi slt, %c127_i8, %in : i8
    %111 = arith.select %110, %c127_i8, %109 : i8
    linalg.yield %111 : i8
} -> tensor<?x8xi8>
