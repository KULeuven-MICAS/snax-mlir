builtin.module {
  func.func @toy(%arg0 : memref<1x18x18x16xi8>, %arg1 : memref<16x3x3x16xi8>, %arg2 : memref<16x10xi8>) -> memref<1x10xi32> {
    %0 = arith.constant 160 : index
    %1 = arith.constant 40 : index
    %2 = arith.constant 256 : index
    %3 = arith.constant 656877351 : i32
    %4 = arith.constant -1838878441 : i32
    %5 = arith.constant 2304 : index
    %6 = arith.constant 5184 : index
    %7 = arith.constant 4096 : index
    %8 = arith.constant 10 : index
    %9 = arith.constant 1035 : i32
    %10 = arith.constant 1014 : i32
    %11 = arith.constant 1 : i5
    %12 = arith.constant 1034 : i32
    %13 = arith.constant 1033 : i32
    %14 = arith.constant 1032 : i32
    %15 = arith.constant 1031 : i32
    %16 = arith.constant 1030 : i32
    %17 = arith.constant 1029 : i32
    %18 = arith.constant 1028 : i32
    %19 = arith.constant 1027 : i32
    %20 = arith.constant 1026 : i32
    %21 = arith.constant 1025 : i32
    %22 = arith.constant 1024 : i32
    %23 = arith.constant 1023 : i32
    %24 = arith.constant 1022 : i32
    %25 = arith.constant 1021 : i32
    %26 = arith.constant 1020 : i32
    %27 = arith.constant 1019 : i32
    %28 = arith.constant 1018 : i32
    %29 = arith.constant 1017 : i32
    %30 = arith.constant 1013 : i32
    %31 = arith.constant 1012 : i32
    %32 = arith.constant 1011 : i32
    %33 = arith.constant 1010 : i32
    %34 = arith.constant 1009 : i32
    %35 = arith.constant 1008 : i32
    %36 = arith.constant 1007 : i32
    %37 = arith.constant 1006 : i32
    %38 = arith.constant 1005 : i32
    %39 = arith.constant 1004 : i32
    %40 = arith.constant 1003 : i32
    %41 = arith.constant 1002 : i32
    %42 = arith.constant 1001 : i32
    %43 = arith.constant 1000 : i32
    %44 = arith.constant 999 : i32
    %45 = arith.constant 998 : i32
    %46 = arith.constant 997 : i32
    %47 = arith.constant 996 : i32
    %48 = arith.constant 995 : i32
    %49 = arith.constant 994 : i32
    %50 = arith.constant 993 : i32
    %51 = arith.constant 992 : i32
    %52 = arith.constant 991 : i32
    %53 = arith.constant 990 : i32
    %54 = arith.constant 989 : i32
    %55 = arith.constant 988 : i32
    %56 = arith.constant 987 : i32
    %57 = arith.constant 986 : i32
    %58 = arith.constant 985 : i32
    %59 = arith.constant 984 : i32
    %60 = arith.constant 983 : i32
    %61 = arith.constant 982 : i32
    %62 = arith.constant 981 : i32
    %63 = arith.constant 980 : i32
    %64 = arith.constant 979 : i32
    %65 = arith.constant 978 : i32
    %66 = arith.constant 977 : i32
    %67 = arith.constant 976 : i32
    %68 = arith.constant 975 : i32
    %69 = arith.constant 974 : i32
    %70 = arith.constant 973 : i32
    %71 = arith.constant 972 : i32
    %72 = arith.constant 971 : i32
    %73 = arith.constant 970 : i32
    %74 = arith.constant 969 : i32
    %75 = arith.constant 968 : i32
    %76 = arith.constant 967 : i32
    %77 = arith.constant 966 : i32
    %78 = arith.constant 965 : i32
    %79 = arith.constant 964 : i32
    %80 = arith.constant 963 : i32
    %81 = arith.constant 962 : i32
    %82 = arith.constant 961 : i32
    %83 = arith.constant 960 : i32
    %84 = arith.constant 1234567890 : i32
    %85 = arith.constant 18 : i32
    %86 = arith.constant -1 : i32
    %87 = arith.constant 2048 : i32
    %88 = arith.constant 268435520 : i32
    %89 = arith.constant 64 : i32
    %90 = arith.constant 32 : i32
    %91 = arith.constant 36 : i32
    %92 = arith.constant 1152 : i32
    %93 = arith.constant 2592 : i32
    %94 = arith.constant 144 : i32
    %95 = arith.constant 16 : i32
    %96 = arith.constant 2 : i32
    %97 = arith.constant 3 : i32
    %98 = arith.constant 8 : i32
    %99 = arith.constant 3 : index
    %100 = arith.constant 18 : index
    %101 = arith.constant 16 : index
    %102 = arith.constant 1 : index
    %103 = arith.constant 1 : i32
    %104 = arith.constant 0 : i32
    %105 = func.call @snax_cluster_core_idx() {"pin_to_constants" = [0 : i32, 1 : i32]} : () -> i32
    %106 = arith.cmpi eq, %105, %104 : i32
    %107 = arith.cmpi eq, %105, %103 : i32
    %108 = arith.constant 64 : index
    %109 = arith.constant 268440064 : i32
    %110 = "llvm.inttoptr"(%109) : (i32) -> !llvm.ptr
    %111 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %112 = "llvm.insertvalue"(%111, %110) <{"position" = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %113 = "llvm.insertvalue"(%112, %110) <{"position" = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %114 = arith.constant 0 : i32
    %115 = "llvm.insertvalue"(%113, %114) <{"position" = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %116 = builtin.unrealized_conversion_cast %102 : index to i32
    %117 = "llvm.insertvalue"(%115, %116) <{"position" = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %118 = builtin.unrealized_conversion_cast %101 : index to i32
    %119 = "llvm.insertvalue"(%117, %118) <{"position" = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %120 = builtin.unrealized_conversion_cast %101 : index to i32
    %121 = "llvm.insertvalue"(%119, %120) <{"position" = array<i64: 3, 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %122 = builtin.unrealized_conversion_cast %101 : index to i32
    %123 = "llvm.insertvalue"(%121, %122) <{"position" = array<i64: 3, 3>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %124 = builtin.unrealized_conversion_cast %123 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)> to memref<1x16x16x16xi8>
    %125 = arith.constant 64 : index
    %126 = arith.constant 268445440 : i32
    %127 = "llvm.inttoptr"(%126) : (i32) -> !llvm.ptr
    %128 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %129 = "llvm.insertvalue"(%128, %127) <{"position" = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %130 = "llvm.insertvalue"(%129, %127) <{"position" = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %131 = arith.constant 0 : i32
    %132 = "llvm.insertvalue"(%130, %131) <{"position" = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %133 = builtin.unrealized_conversion_cast %102 : index to i32
    %134 = "llvm.insertvalue"(%132, %133) <{"position" = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %135 = builtin.unrealized_conversion_cast %100 : index to i32
    %136 = "llvm.insertvalue"(%134, %135) <{"position" = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %137 = builtin.unrealized_conversion_cast %100 : index to i32
    %138 = "llvm.insertvalue"(%136, %137) <{"position" = array<i64: 3, 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %139 = builtin.unrealized_conversion_cast %101 : index to i32
    %140 = "llvm.insertvalue"(%138, %139) <{"position" = array<i64: 3, 3>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %141 = builtin.unrealized_conversion_cast %140 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)> to memref<1x18x18x16xi8>
    %142 = arith.constant 64 : index
    %143 = arith.constant 268447744 : i32
    %144 = "llvm.inttoptr"(%143) : (i32) -> !llvm.ptr
    %145 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %146 = "llvm.insertvalue"(%145, %144) <{"position" = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %147 = "llvm.insertvalue"(%146, %144) <{"position" = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %148 = arith.constant 0 : i32
    %149 = "llvm.insertvalue"(%147, %148) <{"position" = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %150 = builtin.unrealized_conversion_cast %101 : index to i32
    %151 = "llvm.insertvalue"(%149, %150) <{"position" = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %152 = builtin.unrealized_conversion_cast %99 : index to i32
    %153 = "llvm.insertvalue"(%151, %152) <{"position" = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %154 = builtin.unrealized_conversion_cast %99 : index to i32
    %155 = "llvm.insertvalue"(%153, %154) <{"position" = array<i64: 3, 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %156 = builtin.unrealized_conversion_cast %101 : index to i32
    %157 = "llvm.insertvalue"(%155, %156) <{"position" = array<i64: 3, 3>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %158 = builtin.unrealized_conversion_cast %157 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)> to memref<16x3x3x16xi8>
    scf.if %107 {
      %159 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<16x3x3x16xi8>) -> index
      %160 = "memref.extract_aligned_pointer_as_index"(%158) : (memref<16x3x3x16xi8>) -> index
      func.call @snax_dma_1d_transfer(%159, %160, %5) : (index, index, index) -> ()
      %161 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<1x18x18x16xi8>) -> index
      %162 = "memref.extract_aligned_pointer_as_index"(%141) : (memref<1x18x18x16xi8>) -> index
      func.call @snax_dma_1d_transfer(%161, %162, %6) : (index, index, index) -> ()
    }
    func.call @snax_cluster_hw_barrier() : () -> ()
    scf.if %106 {
      %163 = "memref.extract_aligned_pointer_as_index"(%141) : (memref<1x18x18x16xi8>) -> index
      %164 = "memref.extract_aligned_pointer_as_index"(%158) : (memref<16x3x3x16xi8>) -> index
      %165 = "memref.extract_aligned_pointer_as_index"(%124) : (memref<1x16x16x16xi8>) -> index
      %166 = arith.index_cast %163 : index to i32
      "llvm.inline_asm"(%83, %166) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%82, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%81, %98) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%80, %97) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%79, %97) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%78, %96) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%77, %96) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%76, %96) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%75, %95) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%74, %98) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%73, %94) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%72, %93) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%71, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%70, %92) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%69, %98) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      %167 = arith.index_cast %164 : index to i32
      "llvm.inline_asm"(%68, %167) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%67, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%66, %98) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%65, %103) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%64, %91) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%63, %90) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%62, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%61, %89) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%60, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%59, %88) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%58, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%57, %98) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%56, %103) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%55, %96) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%54, %90) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%53, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%52, %87) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%51, %89) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%50, %167) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%49, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%48, %98) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%47, %103) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%46, %96) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%45, %90) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%44, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%43, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%42, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%41, %86) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      %168 = arith.index_cast %165 : index to i32
      "llvm.inline_asm"(%40, %168) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%39, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%38, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%37, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%36, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%35, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%34, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%33, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%32, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%31, %103) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%30, %103) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%29, %85) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%28, %103) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%27, %89) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%26, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%25, %4) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%24, %103) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%23, %3) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%22, %3) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%21, %84) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%20, %84) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%19, %84) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%18, %84) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%17, %84) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%16, %84) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%15, %84) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%14, %84) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%13, %89) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%12, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, rK", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%10, %11) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, K", "has_side_effects"}> : (i32, i5) -> ()
      "llvm.inline_asm"(%9, %11) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, K", "has_side_effects"}> : (i32, i5) -> ()
      "llvm.inline_asm"(%9, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, K", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%9, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, K", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%10, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, K", "has_side_effects"}> : (i32, i32) -> ()
      "llvm.inline_asm"(%10, %104) <{"asm_dialect" = 0 : i64, "asm_string" = "csrw $0, $1", "constraints" = "I, K", "has_side_effects"}> : (i32, i32) -> ()
    }
    func.call @snax_cluster_hw_barrier() : () -> ()
    %169 = arith.constant 64 : index
    %170 = arith.constant 268448000 : i32
    %171 = "llvm.inttoptr"(%170) : (i32) -> !llvm.ptr
    %172 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %173 = "llvm.insertvalue"(%172, %171) <{"position" = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %174 = "llvm.insertvalue"(%173, %171) <{"position" = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %175 = arith.constant 0 : i32
    %176 = "llvm.insertvalue"(%174, %175) <{"position" = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %177 = builtin.unrealized_conversion_cast %102 : index to i32
    %178 = "llvm.insertvalue"(%176, %177) <{"position" = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %179 = builtin.unrealized_conversion_cast %102 : index to i32
    %180 = "llvm.insertvalue"(%178, %179) <{"position" = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %181 = builtin.unrealized_conversion_cast %102 : index to i32
    %182 = "llvm.insertvalue"(%180, %181) <{"position" = array<i64: 3, 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %183 = builtin.unrealized_conversion_cast %101 : index to i32
    %184 = "llvm.insertvalue"(%182, %183) <{"position" = array<i64: 3, 3>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)>
    %185 = builtin.unrealized_conversion_cast %184 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<4 x i32>, !llvm.array<4 x i32>)> to memref<1x1x1x16xi8>
    %186 = arith.constant 64 : index
    %187 = arith.constant 268448256 : i32
    %188 = "llvm.inttoptr"(%187) : (i32) -> !llvm.ptr
    %189 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %190 = "llvm.insertvalue"(%189, %188) <{"position" = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %191 = "llvm.insertvalue"(%190, %188) <{"position" = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %192 = arith.constant 0 : i32
    %193 = "llvm.insertvalue"(%191, %192) <{"position" = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %194 = builtin.unrealized_conversion_cast %101 : index to i32
    %195 = "llvm.insertvalue"(%193, %194) <{"position" = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %196 = builtin.unrealized_conversion_cast %101 : index to i32
    %197 = "llvm.insertvalue"(%195, %196) <{"position" = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %198 = builtin.unrealized_conversion_cast %197 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)> to memref<16x16xi8>
    scf.if %107 {
      %199 = "memref.cast"(%124) : (memref<1x16x16x16xi8>) -> memref<?x?x?x?xi8>
      %200 = "memref.cast"(%198) : (memref<16x16xi8>) -> memref<?x?xi8>
      %201 = "memref.cast"(%185) : (memref<1x1x1x16xi8>) -> memref<?x?x?x?xi8>
      func.call @snax_xdma(%199, %200, %201) : (memref<?x?x?x?xi8>, memref<?x?xi8>, memref<?x?x?x?xi8>) -> ()
    }
    func.call @snax_cluster_hw_barrier() : () -> ()
    %202 = memref.collapse_shape %185 [[0 : i64, 1 : i64, 2 : i64], [3 : i64]] : memref<1x1x1x16xi8> into memref<1x16xi8>
    %203 = arith.constant 64 : index
    %204 = arith.constant 268448512 : i32
    %205 = "llvm.inttoptr"(%204) : (i32) -> !llvm.ptr
    %206 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %207 = "llvm.insertvalue"(%206, %205) <{"position" = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %208 = "llvm.insertvalue"(%207, %205) <{"position" = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %209 = arith.constant 0 : i32
    %210 = "llvm.insertvalue"(%208, %209) <{"position" = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %211 = builtin.unrealized_conversion_cast %102 : index to i32
    %212 = "llvm.insertvalue"(%210, %211) <{"position" = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %213 = builtin.unrealized_conversion_cast %8 : index to i32
    %214 = "llvm.insertvalue"(%212, %213) <{"position" = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %215 = builtin.unrealized_conversion_cast %214 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)> to memref<1x10xi32>
    %216 = arith.constant 64 : index
    %217 = arith.constant 268448768 : i32
    %218 = "llvm.inttoptr"(%217) : (i32) -> !llvm.ptr
    %219 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %220 = "llvm.insertvalue"(%219, %218) <{"position" = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %221 = "llvm.insertvalue"(%220, %218) <{"position" = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %222 = arith.constant 0 : i32
    %223 = "llvm.insertvalue"(%221, %222) <{"position" = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %224 = builtin.unrealized_conversion_cast %101 : index to i32
    %225 = "llvm.insertvalue"(%223, %224) <{"position" = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %226 = builtin.unrealized_conversion_cast %8 : index to i32
    %227 = "llvm.insertvalue"(%225, %226) <{"position" = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
    %228 = builtin.unrealized_conversion_cast %227 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)> to memref<16x10xi8>
    scf.if %107 {
      %229 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<16x10xi8>) -> index
      %230 = "memref.extract_aligned_pointer_as_index"(%228) : (memref<16x10xi8>) -> index
      func.call @snax_dma_1d_transfer(%229, %230, %0) : (index, index, index) -> ()
    }
    func.call @snax_cluster_hw_barrier() : () -> ()
    scf.if %106 {
      linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], library_call = "none"} ins(%202, %228, %104, %104 : memref<1x16xi8>, memref<16x10xi8>, i32, i32) outs(%215 : memref<1x10xi32>) {
      ^0(%231 : i8, %232 : i8, %233 : i32, %234 : i32, %235 : i32):
        %236 = arith.extsi %231 : i8 to i32
        %237 = arith.subi %236, %233 : i32
        %238 = arith.extsi %232 : i8 to i32
        %239 = arith.subi %238, %234 : i32
        %240 = arith.muli %237, %239 : i32
        %241 = arith.addi %235, %240 : i32
        linalg.yield %241 : i32
      }
    }
    func.call @snax_cluster_hw_barrier() : () -> ()
    func.return %215 : memref<1x10xi32>
  }
  func.func private @snax_cluster_core_idx() -> i32
  func.func private @snax_dma_1d_transfer(index, index, index) -> ()
  func.func private @snax_cluster_hw_barrier() -> ()
  func.func private @snax_xdma(memref<?x?x?x?xi8>, memref<?x?xi8>, memref<?x?x?x?xi8>) -> ()
}

