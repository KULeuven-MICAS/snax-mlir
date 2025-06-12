// RUN: snax-opt --split-input-file -p convert-memref-to-arith %s | filecheck %s

%0 = "test.op"() : () -> (memref<16x16xi8, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>>)
%1 = "test.op"() : () -> (index)
%2 = "test.op"() : () -> (index)
%3 = memref.subview %0[%1, 0] [8, 16] [1, 1] : memref<16x16xi8, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>> to memref<8x16xi8, #tsl.tsl<[8] -> (8), [2, 8] -> (64, 1)>>
%4 = "memref.extract_aligned_pointer_as_index"(%3) : (memref<8x16xi8, #tsl.tsl<[8] -> (8), [2, 8] -> (64, 1)>>) -> index
"test.op"(%4) : (index) -> ()

%5 = memref.subview %0[%1, %2] [8, 16] [1, 1] : memref<16x16xi8, #tsl.tsl<[2, 8] -> (128, 8), [2, 8] -> (64, 1)>> to memref<8x8xi8, #tsl.tsl<[8] -> (8), [8] -> (1)>>
%6 = "memref.extract_aligned_pointer_as_index"(%5) : (memref<8x8xi8, #tsl.tsl<[8] -> (8), [8] -> (1)>>) -> index
"test.op"(%6) : (index) -> ()

