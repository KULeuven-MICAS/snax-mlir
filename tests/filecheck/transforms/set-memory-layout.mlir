// RUN: ./compiler/snax-opt --split-input-file %s -p set-memory-layout --print-op-generic | filecheck %s
// RUN: ./compiler/snax-opt --split-input-file %s -p set-memory-layout{gemm_layout=banked} --print-op-generic | filecheck %s --check-prefix=BANKED

builtin.module {
  func.func @mnist(%arg0 : memref<?x128xi8, 1 : i32>, %arg1 : memref<128x128xi8, 1 : i32>, %arg2 : memref<?x128xi32, 1 : i32>) -> memref<?x128xi32, 1 : i32> {
    %0 = arith.constant 1 : i32
    %1 = arith.constant 1 : i32
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], library_call = "snax_gemmx"} ins(%arg0, %arg1, %0, %1 : memref<?x128xi8, 1 : i32>, memref<128x128xi8, 1 : i32>, i32, i32) outs(%arg2 : memref<?x128xi32, 1 : i32>) {
    ^0(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
      %2 = kernel.qmac %arg3, %arg4 zp_lhs : %arg5 zp_rhs : %arg6 : i8, i8, i32, i32 -> i32
      linalg.yield %2 : i32
    }
    func.return %arg2 : memref<?x128xi32, 1 : i32>
  }
}


//CHECK:       %2 = "snax.layout_cast"(%arg0) : (memref<?x128xi8, 1 : i32>) -> memref<?x128xi8, #tsl.tsl<[?, 8] -> (1024, 8), [16, 8] -> (64, 1)>, 1 : i32>
//CHECK-NEXT:  %3 = "snax.layout_cast"(%arg1) : (memref<128x128xi8, 1 : i32>) -> memref<128x128xi8, #tsl.tsl<[16, 8] -> (64, 1), [16, 8] -> (1024, 8)>, 1 : i32>
//CHECK-NEXT:  %4 = "snax.layout_cast"(%arg2) : (memref<?x128xi32, 1 : i32>) -> memref<?x128xi32, #tsl.tsl<[?, 8] -> (1024, 8), [16, 8] -> (64, 1)>, 1 : i32>

//BANKED:       %2 = "snax.layout_cast"(%arg0) : (memref<?x128xi8, 1 : i32>) -> memref<?x128xi8, #tsl.tsl<[?, 8] -> (4096, 8), [16, 8] -> (256, 1)>, 1 : i32>
//BANKED-NEXT:  %3 = "snax.layout_cast"(%arg1) : (memref<128x128xi8, 1 : i32>) -> memref<128x128xi8, #tsl.tsl<[16, 8] -> (256, 1), [16, 8] -> (4096, 8)>, 1 : i32>
//BANKED-NEXT:  %4 = "snax.layout_cast"(%arg2) : (memref<?x128xi32, 1 : i32>) -> memref<?x128xi32, #tsl.tsl<[?, 8] -> (1024, 8), [16, 8] -> (64, 1)>, 1 : i32>

// -----

func.func @gemm(%arg0 : memref<16x16xi8, "L1">, %arg1 : memref<16x16xi8, "L1">, %arg2 : memref<16x16xi32, "L1">) -> () {
  %0 = arith.constant 0 : i32
  %1 = arith.constant 0 : i32
  "stream.streaming_region"(%arg0, %arg1, %arg2) <{"accelerator" = "snax_gemmx", "operandSegmentSizes" = array<i32: 2, 1>, "patterns" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
  ^0(%arg3 : !stream.stream<i8>, %arg4 : !stream.stream<i8>, %arg5 : !stream.stream<i32>):
    %5 = "stream.generic"(%arg3, %arg4, %0, %1) <{"library_call" = "snax_gemmx"}> ({
    ^1(%arg6 : i8, %arg7 : i8, %arg8 : i32, %arg9 : i32, %arg10 : i32):
      %6 = kernel.qmac %arg6, %arg7 zp_lhs : %arg8 zp_rhs : %arg9 : i8, i8, i32, i32 -> i32
      stream.yield %6 : i32
    }) : (!stream.stream<i8>, !stream.stream<i8>, i32, i32) -> !stream.stream<i32>
    stream.yield %5 : !stream.stream<i32>
  }) : (memref<16x16xi8, "L1">, memref<16x16xi8, "L1">, memref<16x16xi32, "L1">) -> ()
  func.return
}

// CHECK:      %2 = "snax.layout_cast"(%arg0) : (memref<16x16xi8, "L1">) -> memref<16x16xi8, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>, "L1">
// CHECK-NEXT: %3 = "snax.layout_cast"(%arg1) : (memref<16x16xi8, "L1">) -> memref<16x16xi8, #tsl.tsl<[2, 8] -> (64, 1), [2, 8] -> (128, 8)>, "L1">
// CHECK-NEXT: %4 = "snax.layout_cast"(%arg2) : (memref<16x16xi32, "L1">) -> memref<16x16xi32, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>, "L1">

// BANKED:       %2 = "snax.layout_cast"(%arg0) : (memref<16x16xi8, "L1">) -> memref<16x16xi8, #tsl.tsl<[2, 8] -> (512, 8), [2, 8] -> (256, 1)>, "L1">
// BANKED-NEXT:  %3 = "snax.layout_cast"(%arg1) : (memref<16x16xi8, "L1">) -> memref<16x16xi8, #tsl.tsl<[2, 8] -> (256, 1), [2, 8] -> (512, 8)>, "L1">
// BANKED-NEXT:  %4 = "snax.layout_cast"(%arg2) : (memref<16x16xi32, "L1">) -> memref<16x16xi32, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>, "L1">
