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
  "func.func"() <{"sym_name" = "transform_copy", "function_type" = (memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 0 : i32>, memref<8x8xi32, #tsl.tsl<[2, 4] -> (64, 4), [2, 4] -> (128, 16)>, 1 : i32>) -> (), "sym_visibility" = "public"}> ({
  ^0(%arg0 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 0 : i32>, %arg1 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (64, 4), [2, 4] -> (128, 16)>, 1 : i32>):
    "memref.copy"(%arg0, %arg1) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 0 : i32>, memref<8x8xi32, #tsl.tsl<[2, 4] -> (64, 4), [2, 4] -> (128, 16)>, 1 : i32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   "func.func"() <{"sym_name" = "transform_copy", "function_type" = (memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 0 : i32>, memref<8x8xi32, #tsl.tsl<[2, 4] -> (64, 4), [2, 4] -> (128, 16)>, 1 : i32>) -> (), "sym_visibility" = "public"}> ({
//CHECK-NEXT:   ^0(%arg0 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 0 : i32>, %arg1 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (64, 4), [2, 4] -> (128, 16)>, 1 : i32>):
//CHECK-NEXT:     %0 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 0 : i32>) -> index
//CHECK-NEXT:     %1 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (64, 4), [2, 4] -> (128, 16)>, 1 : i32>) -> index
//CHECK-NEXT:     %2 = "arith.constant"() <{"value" = 2 : index}> : () -> index
//CHECK-NEXT:     %3 = "arith.constant"() <{"value" = 4 : index}> : () -> index
//CHECK-NEXT:     %4 = "arith.constant"() <{"value" = 2 : index}> : () -> index
//CHECK-NEXT:     %5 = "arith.constant"() <{"value" = 4 : index}> : () -> index
//CHECK-NEXT:     %6 = "arith.constant"() <{"value" = 128 : index}> : () -> index
//CHECK-NEXT:     %7 = "arith.constant"() <{"value" = 32 : index}> : () -> index
//CHECK-NEXT:     %8 = "arith.constant"() <{"value" = 128 : index}> : () -> index
//CHECK-NEXT:     %9 = "arith.constant"() <{"value" = 4 : index}> : () -> index
//CHECK-NEXT:     %10 = "arith.constant"() <{"value" = 16 : index}> : () -> index
//CHECK-NEXT:     %11 = "arith.constant"() <{"value" = 128 : index}> : () -> index
//CHECK-NEXT:     %12 = "arith.constant"() <{"value" = 16 : index}> : () -> index
//CHECK-NEXT:     %13 = "arith.constant"() <{"value" = 128 : index}> : () -> index
//CHECK-NEXT:     %14 = "arith.constant"() <{"value" = 4 : index}> : () -> index
//CHECK-NEXT:     %15 = "arith.constant"() <{"value" = 64 : index}> : () -> index
//CHECK-NEXT:     %16 = "arith.constant"() <{"value" = 16 : index}> : () -> index
//CHECK-NEXT:     %17 = "arith.constant"() <{"value" = 0 : index}> : () -> index
//CHECK-NEXT:     %18 = "arith.constant"() <{"value" = 1 : index}> : () -> index
//CHECK-NEXT:     "scf.for"(%17, %2, %18) ({
//CHECK-NEXT:     ^1(%19 : index):
//CHECK-NEXT:       %20 = "arith.muli"(%19, %10) : (index, index) -> index
//CHECK-NEXT:       %21 = "arith.addi"(%0, %20) : (index, index) -> index
//CHECK-NEXT:       %22 = "arith.muli"(%19, %15) : (index, index) -> index
//CHECK-NEXT:       %23 = "arith.addi"(%1, %22) : (index, index) -> index
//CHECK-NEXT:       "scf.for"(%17, %4, %18) ({
//CHECK-NEXT:       ^2(%24 : index):
//CHECK-NEXT:         %25 = "arith.muli"(%24, %8) : (index, index) -> index
//CHECK-NEXT:         %26 = "arith.addi"(%21, %25) : (index, index) -> index
//CHECK-NEXT:         %27 = "arith.muli"(%24, %13) : (index, index) -> index
//CHECK-NEXT:         %28 = "arith.addi"(%23, %27) : (index, index) -> index
//CHECK-NEXT:         "func.call"(%26, %28, %16, %7, %12, %5) <{"callee" = @snax_dma_2d_transfer}> : (index, index, index, index, index, index) -> ()
//CHECK-NEXT:         "scf.yield"() : () -> ()
//CHECK-NEXT:       }) : (index, index, index) -> ()
//CHECK-NEXT:       "scf.yield"() : () -> ()
//CHECK-NEXT:     }) : (index, index, index) -> ()
//CHECK-NEXT:     "func.return"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT:   "func.func"() <{"sym_name" = "snax_dma_2d_transfer", "function_type" = (index, index, index, index, index, index) -> (), "sym_visibility" = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()
