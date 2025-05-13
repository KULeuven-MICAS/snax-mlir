// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_SINGLETRIP

%0 = "test.op"() : () -> memref<8x8xi32, strided<[1, 8]>, "L1">
// CHECK:   %0 = "test.op"() : () -> memref<8x8xi32, strided<[1, 8]>, "L1">

%1 = "snax.layout_cast"(%0) : (memref<8x8xi32, strided<[1, 8]>, "L1">) -> memref<8x8xi32, strided<[1, 16]>, "L1">
// CHECK-NEXT:   %1 = "snax.layout_cast"(%0) : (memref<8x8xi32, strided<[1, 8]>, "L1">) -> memref<8x8xi32, strided<[1, 16]>, "L1">

"snax.mcycle"() : () -> ()
// CHECK-NEXT:   "snax.mcycle"() : () -> ()

%2 = "test.op"() : () -> index
%3 = "snax.alloc"(%2, %2, %2) <{memory_space = "L3", alignment = 64 : i32}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %2 = "test.op"() : () -> index
// CHECK-NEXT:   %3 = "snax.alloc"(%2, %2, %2) <{memory_space = "L3", alignment = 64 : i32}> : (index, index, index) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>

// Test streamer config attribute:
"test.op"() {streamer_config = #snax.streamer_config<r[temp=n-n-n, spat=1-2-3], r[temp=n-r-i, spat=4-5-6], w[temp=i-i-i, spat=7-8]> } : () -> ()
// CHECK-NEXT: #snax.streamer_config<r[temp=n-n-n, spat=1-2-3], r[temp=n-r-i, spat=4-5-6], w[temp=i-i-i, spat=7-8]>
"test.op"() {streamer_config = #snax.streamer_config <r[opts=c-t, temp=n-n-n, spat=1-2-3], r[opts=c, temp=n-r-i, spat=2-4-6], w[opts=t, temp=i-i-i, spat=2-3]> } : () -> ()
// CHECK-NEXT: #snax.streamer_config<r[opts={{c-t|t-c}}, temp=n-n-n, spat=1-2-3], r[opts=c, temp=n-r-i, spat=2-4-6], w[opts=t, temp=i-i-i, spat=2-3]>
