// RUN: ./compiler/snax-opt --split-input-file %s -p set-memory-layout --print-op-generic | filecheck %s

"builtin.module"() ({
  "func.func"() <{"sym_name" = "mnist", "function_type" = (memref<?x128xi8, 1 : i32>, memref<128x128xi8, 1 : i32>, memref<?x128xi32, 1 : i32>) -> memref<?x128xi32, 1 : i32>}> ({
  ^0(%arg0 : memref<?x128xi8, 1 : i32>, %arg1 : memref<128x128xi8, 1 : i32>, %arg2 : memref<?x128xi32, 1 : i32>):
    %0 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
    %1 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
    "linalg.generic"(%arg0, %arg1, %0, %1, %arg2) <{"indexing_maps" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], "iterator_types" = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], "operandSegmentSizes" = array<i32: 4, 1>, "library_call" = "snax_gemm"}> ({
    ^1(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
      %2 = "arith.extsi"(%arg3) : (i8) -> i32
      %3 = "arith.subi"(%2, %arg5) : (i32, i32) -> i32
      %4 = "arith.extsi"(%arg4) : (i8) -> i32
      %5 = "arith.subi"(%4, %arg6) : (i32, i32) -> i32
      %6 = "arith.muli"(%3, %5) : (i32, i32) -> i32
      %7 = "arith.addi"(%arg7, %6) : (i32, i32) -> i32
      "linalg.yield"(%7) : (i32) -> ()
    }) : (memref<?x128xi8, 1 : i32>, memref<128x128xi8, 1 : i32>, i32, i32, memref<?x128xi32, 1 : i32>) -> ()
    "func.return"(%arg2) : (memref<?x128xi32, 1 : i32>) -> ()
  }) : () -> ()
}) : () -> ()


//CHECK:       %2 = "snax.layout_cast"(%arg0) : (memref<?x128xi8, 1 : i32>) -> memref<?x128xi8, #tsl.tsl<[?, 8] -> (4096, 8), [16, 8] -> (256, 1)>, 1 : i32>
//CHECK-NEXT:  %3 = "snax.layout_cast"(%arg1) : (memref<128x128xi8, 1 : i32>) -> memref<128x128xi8, #tsl.tsl<[16, 8] -> (256, 1), [16, 8] -> (4096, 8)>, 1 : i32>
//CHECK-NEXT:  %4 = "snax.layout_cast"(%arg2) : (memref<?x128xi32, 1 : i32>) -> memref<?x128xi32, #tsl.tsl<[?, 8] -> (1024, 8), [16, 8] -> (64, 1)>, 1 : i32>
