// RUN: ./compiler/snax-opt --split-input-file %s -p snax-copy-to-dma --print-op-generic | filecheck %s

"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<?xi32>, memref<?xi32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>):
    "memref.copy"(%arg0, %arg1) : (memref<?xi32>, memref<?xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<?xi32>, memref<?xi32>) -> (), "sym_visibility" = "public"}> ({
//CHECK-NEXT:   ^0(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>):
//CHECK-NEXT:     %0 = "arith.constant"() <{"value" = 0 : index}> : () -> index
//CHECK-NEXT:     %1 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index
//CHECK-NEXT:     %2 = "arith.constant"() <{"value" = 4 : index}> : () -> index
//CHECK-NEXT:     %3 = "arith.muli"(%1, %2) : (index, index) -> index
//CHECK-NEXT:     %4 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
//CHECK-NEXT:     %5 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
//CHECK-NEXT:     "func.call"(%4, %5, %3) <{"callee" = @snax_dma_1d_transfer}> : (index, index, index) -> ()
//CHECK-NEXT:     "func.return"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snax_dma_1d_transfer", "function_type" = (index, index, index) -> (), "sym_visibility" = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<?x?xi32>, memref<?x?xi32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<?x?xi32>, %arg1 : memref<?x?xi32>):
    "memref.copy"(%arg0, %arg1) : (memref<?x?xi32>, memref<?x?xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<?x?xi32>, memref<?x?xi32>) -> (), "sym_visibility" = "public"}> ({
//CHECK-NEXT:   ^0(%arg0 : memref<?x?xi32>, %arg1 : memref<?x?xi32>):
//CHECK-NEXT:     %0 = "arith.constant"() <{"value" = 0 : index}> : () -> index
//CHECK-NEXT:     %1 = "memref.dim"(%arg0, %0) : (memref<?x?xi32>, index) -> index
//CHECK-NEXT:     %2 = "arith.constant"() <{"value" = 1 : index}> : () -> index
//CHECK-NEXT:     %3 = "memref.dim"(%arg0, %2) : (memref<?x?xi32>, index) -> index
//CHECK-NEXT:     %4 = "arith.muli"(%1, %3) : (index, index) -> index
//CHECK-NEXT:     %5 = "arith.constant"() <{"value" = 4 : index}> : () -> index
//CHECK-NEXT:     %6 = "arith.muli"(%4, %5) : (index, index) -> index
//CHECK-NEXT:     %7 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?x?xi32>) -> index
//CHECK-NEXT:     %8 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?x?xi32>) -> index
//CHECK-NEXT:     "func.call"(%7, %8, %6) <{"callee" = @snax_dma_1d_transfer}> : (index, index, index) -> ()
//CHECK-NEXT:     "func.return"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snax_dma_1d_transfer", "function_type" = (index, index, index) -> (), "sym_visibility" = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<5x5xi32, strided<[10, 1]>>, memref<5x5xi32, strided<[20, 1]>>) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<5x5xi32, strided<[10, 1]>>, %arg1 : memref<5x5xi32, strided<[20, 1]>>):
    "memref.copy"(%arg0, %arg1) : (memref<5x5xi32, strided<[10, 1]>>, memref<5x5xi32, strided<[20, 1]>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   "func.func"() <{"sym_name" = "simple_mult", "function_type" = (memref<5x5xi32, strided<[10, 1]>>, memref<5x5xi32, strided<[20, 1]>>) -> (), "sym_visibility" = "public"}> ({
//CHECK-NEXT:   ^0(%arg0 : memref<5x5xi32, strided<[10, 1]>>, %arg1 : memref<5x5xi32, strided<[20, 1]>>):
//CHECK-NEXT:     %0 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<5x5xi32, strided<[10, 1]>>) -> index
//CHECK-NEXT:     %1 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<5x5xi32, strided<[20, 1]>>) -> index
//CHECK-NEXT:     %2 = "arith.constant"() <{"value" = 0 : index}> : () -> index
//CHECK-NEXT:     %3 = "memref.dim"(%arg0, %2) : (memref<5x5xi32, strided<[10, 1]>>, index) -> index
//CHECK-NEXT:     %4 = "arith.constant"() <{"value" = 1 : index}> : () -> index
//CHECK-NEXT:     %5 = "memref.dim"(%arg0, %4) : (memref<5x5xi32, strided<[10, 1]>>, index) -> index
//CHECK-NEXT:     %6 = "arith.constant"() <{"value" = 5 : index}> : () -> index
//CHECK-NEXT:     %7 = "arith.constant"() <{"value" = 5 : index}> : () -> index
//CHECK-NEXT:     %8 = "arith.constant"() <{"value" = 10 : index}> : () -> index
//CHECK-NEXT:     %9 = "arith.constant"() <{"value" = 1 : index}> : () -> index
//CHECK-NEXT:     %10 = "arith.constant"() <{"value" = 10 : index}> : () -> index
//CHECK-NEXT:     %11 = "arith.constant"() <{"value" = 20 : index}> : () -> index
//CHECK-NEXT:     %12 = "arith.constant"() <{"value" = 1 : index}> : () -> index
//CHECK-NEXT:     %13 = "arith.constant"() <{"value" = 20 : index}> : () -> index
//CHECK-NEXT:     %14 = "arith.constant"() <{"value" = 1 : index}> : () -> index
//CHECK-NEXT:     %15 = "arith.constant"() <{"value" = 0 : index}> : () -> index
//CHECK-NEXT:     %16 = "arith.constant"() <{"value" = 1 : index}> : () -> index
//CHECK-NEXT:     "scf.for"(%15, %7, %16) ({
//CHECK-NEXT:     ^1(%17 : index):
//CHECK-NEXT:       %18 = "arith.muli"(%17, %9) : (index, index) -> index
//CHECK-NEXT:       %19 = "arith.addi"(%0, %18) : (index, index) -> index
//CHECK-NEXT:       %20 = "arith.muli"(%17, %12) : (index, index) -> index
//CHECK-NEXT:       %21 = "arith.addi"(%1, %20) : (index, index) -> index
//CHECK-NEXT:       "func.call"(%19, %21, %14, %10, %13, %6) <{"callee" = @snax_dma_2d_transfer}> : (index, index, index, index, index, index) -> ()
//CHECK-NEXT:       "scf.yield"() : () -> ()
//CHECK-NEXT:     }) : (index, index, index) -> ()
//CHECK-NEXT:     "func.return"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snax_dma_2d_transfer", "function_type" = (index, index, index, index, index, index) -> (), "sym_visibility" = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  "func.func"() <{"sym_name" = "transform_copy", "function_type" = (memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 0 : i32>, memref<8x8xi32, #tsl.tsl<[2, 4] -> (64, 4), [2, 4] -> (128, 16)>, 1 : i32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 0 : i32>, %arg1 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (64, 4), [2, 4] -> (128, 16)>, 1 : i32>):
    "memref.copy"(%arg0, %arg1) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 0 : i32>, memref<8x8xi32, #tsl.tsl<[2, 4] -> (64, 4), [2, 4] -> (128, 16)>, 1 : i32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   "func.func"() <{"sym_name" = "transform_copy", "function_type" = (memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 0 : i32>, memref<8x8xi32, #tsl.tsl<[2, 4] -> (64, 4), [2, 4] -> (128, 16)>, 1 : i32>) -> (), "sym_visibility" = "public"}> ({
// CHECK-NEXT:   ^0(%arg0 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 0 : i32>, %arg1 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (64, 4), [2, 4] -> (128, 16)>, 1 : i32>):
// CHECK-NEXT:     %0 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 0 : i32>) -> index
// CHECK-NEXT:     %1 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (64, 4), [2, 4] -> (128, 16)>, 1 : i32>) -> index
// CHECK-NEXT:     %2 = "arith.constant"() <{"value" = 0 : index}> : () -> index
// CHECK-NEXT:     %3 = "memref.dim"(%arg0, %2) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 0 : i32>, index) -> index
// CHECK-NEXT:     %4 = "arith.constant"() <{"value" = 1 : index}> : () -> index
// CHECK-NEXT:     %5 = "memref.dim"(%arg0, %4) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 0 : i32>, index) -> index
// CHECK-NEXT:     %6 = "arith.constant"() <{"value" = 2 : index}> : () -> index
// CHECK-NEXT:     %7 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:     %8 = "arith.constant"() <{"value" = 2 : index}> : () -> index
// CHECK-NEXT:     %9 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:     %10 = "arith.constant"() <{"value" = 128 : index}> : () -> index
// CHECK-NEXT:     %11 = "arith.constant"() <{"value" = 32 : index}> : () -> index
// CHECK-NEXT:     %12 = "arith.constant"() <{"value" = 128 : index}> : () -> index
// CHECK-NEXT:     %13 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:     %14 = "arith.constant"() <{"value" = 16 : index}> : () -> index
// CHECK-NEXT:     %15 = "arith.constant"() <{"value" = 128 : index}> : () -> index
// CHECK-NEXT:     %16 = "arith.constant"() <{"value" = 16 : index}> : () -> index
// CHECK-NEXT:     %17 = "arith.constant"() <{"value" = 128 : index}> : () -> index
// CHECK-NEXT:     %18 = "arith.constant"() <{"value" = 4 : index}> : () -> index
// CHECK-NEXT:     %19 = "arith.constant"() <{"value" = 64 : index}> : () -> index
// CHECK-NEXT:     %20 = "arith.constant"() <{"value" = 16 : index}> : () -> index
// CHECK-NEXT:     %21 = "arith.constant"() <{"value" = 0 : index}> : () -> index
// CHECK-NEXT:     %22 = "arith.constant"() <{"value" = 1 : index}> : () -> index
// CHECK-NEXT:     "scf.for"(%21, %6, %22) ({
// CHECK-NEXT:     ^1(%23 : index):
// CHECK-NEXT:       %24 = "arith.muli"(%23, %14) : (index, index) -> index
// CHECK-NEXT:       %25 = "arith.addi"(%0, %24) : (index, index) -> index
// CHECK-NEXT:       %26 = "arith.muli"(%23, %19) : (index, index) -> index
// CHECK-NEXT:       %27 = "arith.addi"(%1, %26) : (index, index) -> index
// CHECK-NEXT:       "scf.for"(%21, %8, %22) ({
// CHECK-NEXT:       ^2(%28 : index):
// CHECK-NEXT:         %29 = "arith.muli"(%28, %12) : (index, index) -> index
// CHECK-NEXT:         %30 = "arith.addi"(%25, %29) : (index, index) -> index
// CHECK-NEXT:         %31 = "arith.muli"(%28, %17) : (index, index) -> index
// CHECK-NEXT:         %32 = "arith.addi"(%27, %31) : (index, index) -> index
// CHECK-NEXT:         "func.call"(%30, %32, %20, %11, %16, %9) <{"callee" = @snax_dma_2d_transfer}> : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:         "scf.yield"() : () -> ()
// CHECK-NEXT:       }) : (index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) : (index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT:   "func.func"() <{"sym_name" = "snax_dma_2d_transfer", "function_type" = (index, index, index, index, index, index) -> (), "sym_visibility" = "private"}> ({
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()
