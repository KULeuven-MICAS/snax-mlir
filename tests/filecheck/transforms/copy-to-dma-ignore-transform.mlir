// RUN: snax-opt --split-input-file %s -p snax-copy-to-dma{test-ignore-transform=true} | filecheck %s

// a very complex transformation:
"builtin.module"() ({
  "func.func"() <{"sym_name" = "transform_copy", "function_type" = (memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">, memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 1), [2, 4] -> (32, 4)>, "L1">) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">, %arg1 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 1), [2, 4] -> (32, 4)>, "L1">):
    "memref.copy"(%arg0, %arg1) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (4, 1), [2, 4] -> (32, 8)>, "L3">, memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 1), [2, 4] -> (32, 4)>, "L1">) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// but a very simple 1d transfer applied:

// CHECK: func.call @snax_dma_1d_transfer(%0, %1, %{{.*}}) : (index, index, index) -> ()
