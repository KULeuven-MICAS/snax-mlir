// RUN: snax-opt --split-input-file %s -p set-memory-layout{tiled=false} --print-op-generic | filecheck %s
// RUN: snax-opt --split-input-file %s -p set-memory-layout{tiled=true} --print-op-generic | filecheck %s --check-prefix=TILED

func.func @gemm(%arg0 : memref<16x16xi8, "L1">, %arg1 : memref<16x16xi8, "L1">, %arg2 : memref<16x16xi32, "L1">) {
	%0 = arith.constant 0 : i32
	%1 = arith.constant 0 : i32
	"dart.schedule"(%arg0, %arg1, %arg2) <{patterns = [affine_map<(d0, d1, d2, d3, d4, d5) -> (((d0 * 8) + d3), ((d2 * 8) + d5))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (((d2 * 8) + d5), ((d1 * 8) + d4))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (((d0 * 8) + d3), ((d1 * 8) + d4))>], accelerator = "snax_gemmx", tiles = [[]], bounds = [2 : index, 2 : index, 2 : index, 8 : index, 8 : index, 8 : index], operandSegmentSizes = array<i32: 2, 1>}> ({
	^0(%arg3 : !dart.stream<i8>, %arg4 : !dart.stream<i8>, %arg5 : !dart.stream<i32>):
		%2 = "dart.generic"(%arg3, %arg4, %0, %1) <{library_call = "snax_gemmx"}> ({
		^1(%arg6 : i8, %arg7 : i8, %arg8 : i32, %arg9 : i32, %arg10 : i32):
			%3 = kernel.qmac %arg6, %arg7 zp_lhs : %arg8 zp_rhs : %arg9 : i8, i8, i32, i32 -> i32
			dart.yield %3 : i32
		}) : (!dart.stream<i8>, !dart.stream<i8>, i32, i32) -> !dart.stream<i32>
		dart.yield %2 : !dart.stream<i32>
	}) : (memref<16x16xi8, "L1">, memref<16x16xi8, "L1">, memref<16x16xi32, "L1">) -> ()
	func.return
}

// TILED:      %2 = "snax.layout_cast"(%arg0) : (memref<16x16xi8, "L1">) -> memref<16x16xi8, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>, "L1">
// TILED-NEXT: %3 = "snax.layout_cast"(%arg1) : (memref<16x16xi8, "L1">) -> memref<16x16xi8, #tsl.tsl<[2, 8] -> (64, 1), [2, 8] -> (128, 8)>, "L1">
// TILED-NEXT: %4 = "snax.layout_cast"(%arg2) : (memref<16x16xi32, "L1">) -> memref<16x16xi32, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>, "L1">

// CHECK:      %2 = "snax.layout_cast"(%arg0) : (memref<16x16xi8, "L1">) -> memref<16x16xi8, #tsl.tsl<[16] -> (16), [16] -> (1)>, "L1">
// CHECK-NEXT: %3 = "snax.layout_cast"(%arg1) : (memref<16x16xi8, "L1">) -> memref<16x16xi8, #tsl.tsl<[16] -> (1), [16] -> (16)>, "L1">
// CHECK-NEXT: %4 = "snax.layout_cast"(%arg2) : (memref<16x16xi32, "L1">) -> memref<16x16xi32, #tsl.tsl<[16] -> (16), [16] -> (1)>, "L1">

// -----

func.func @snax_main() -> () {
  %4 = memref.alloc() : memref<1x16x18x18xi8, "L1">
  %5 = memref.alloc() : memref<16x16x3x3xi8, "L1">
  %6 = memref.alloc() : memref<1x16x16x16xi32, "L1">
  "dart.schedule"(%4, %5, %6) <{patterns = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d3, ((d6 * 8) + d9), (((d1 * 8) + d5) + d7), (d0 + d4))>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (((d2 * 8) + d8), ((d6 * 8) + d9), d5, d4)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d3, ((d2 * 8) + d8), ((d1 * 8) + d7), d0)>], accelerator = "snax_gemmx", tiles = [[]], bounds = [16 : index, 2 : index, 2 : index, 1 : index, 3 : index, 3 : index, 2 : index, 8 : index, 8 : index, 8 : index], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^0(%arg0 : !dart.stream<i8>, %arg1 : !dart.stream<i8>, %arg2 : !dart.stream<i32>):
    %7 = "dart.generic"(%arg0, %arg1) <{library_call = "snax_gemmx"}> ({
    ^1(%arg3 : i8, %arg4 : i8, %arg5 : i32):
      %8 = kernel.mac %arg3, %arg4 : i8, i8 -> i32
      dart.yield %8 : i32
    }) : (!dart.stream<i8>, !dart.stream<i8>) -> !dart.stream<i32>
    dart.yield %7 : !dart.stream<i32>
  }) : (memref<1x16x18x18xi8, "L1">, memref<16x16x3x3xi8, "L1">, memref<1x16x16x16xi32, "L1">) -> ()
  func.return
}

// TILED:      %3 = "snax.layout_cast"(%0) : (memref<1x16x18x18xi8, "L1">) -> memref<1x16x18x18xi8, #tsl.tsl<[1] -> (5184), [2, 8] -> (144, 1), [18] -> (8), [18] -> (288)>, "L1">
// TILED-NEXT: %4 = "snax.layout_cast"(%1) : (memref<16x16x3x3xi8, "L1">) -> memref<16x16x3x3xi8, #tsl.tsl<[2, 8] -> (1152, 8), [2, 8] -> (64, 1), [3] -> (128), [3] -> (384)>, "L1">
// TILED-NEXT: %5 = "snax.layout_cast"(%2) : (memref<1x16x16x16xi32, "L1">) -> memref<1x16x16x16xi32, #tsl.tsl<[1] -> (64), [2, 8] -> (64, 1), [2, 8] -> (128, 8), [16] -> (256)>, "L1">

// CHECK:      %3 = "snax.layout_cast"(%0) : (memref<1x16x18x18xi8, "L1">) -> memref<1x16x18x18xi8, #tsl.tsl<[1] -> (5184), [16] -> (1), [18] -> (16), [18] -> (288)>, "L1">
// CHECK-NEXT: %4 = "snax.layout_cast"(%1) : (memref<16x16x3x3xi8, "L1">) -> memref<16x16x3x3xi8, #tsl.tsl<[16] -> (16), [16] -> (1), [3] -> (256), [3] -> (768)>, "L1">
// CHECK-NEXT: %5 = "snax.layout_cast"(%2) : (memref<1x16x16x16xi32, "L1">) -> memref<1x16x16x16xi32, #tsl.tsl<[1] -> (256), [16] -> (1), [16] -> (16), [16] -> (256)>, "L1">
