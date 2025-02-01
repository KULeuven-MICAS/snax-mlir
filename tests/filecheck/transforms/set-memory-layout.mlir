// RUN: ./compiler/snax-opt --split-input-file %s -p set-memory-layout{tiled=false} --print-op-generic | filecheck %s
// RUN: ./compiler/snax-opt --split-input-file %s -p set-memory-layout{tiled=true} --print-op-generic | filecheck %s --check-prefix=TILED

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
