builtin.module {
  "llvm.func"() <{CConv = #llvm.cconv<ccc>, function_type = !llvm.func<!llvm.ptr (i64)>, linkage = #llvm.linkage<"external">, sym_name = "malloc", visibility_ = 0 : i64}> ({
  }) : () -> ()
  "llvm.func"() <{CConv = #llvm.cconv<ccc>, function_type = !llvm.func<void (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64)>, linkage = #llvm.linkage<"external">, sym_name = "rescale_down", visibility_ = 0 : i64}> ({
  ^0(%arg2 : !llvm.ptr, %arg3 : !llvm.ptr, %arg4 : i64, %arg5 : i64, %arg6 : i64, %arg7 : !llvm.ptr, %arg8 : !llvm.ptr, %arg9 : i64, %arg10 : i64, %arg11 : i64):
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1073741824) : i64
    %3 = llvm.mlir.constant(-1073741824) : i64
    %4 = llvm.mlir.constant(-128 : i32) : i32
    %5 = llvm.mlir.constant(127 : i32) : i32
    %6 = llvm.mlir.constant(1140768826) : i64
    %7 = llvm.mlir.constant(48) : i64
    %8 = llvm.mlir.constant(140737488355328) : i64
    %9 = llvm.mlir.constant(0) : i64
    %10 = llvm.mlir.constant(64) : i64
    %11 = llvm.mlir.constant(1) : i64
    %12 = "llvm.getelementptr"(%0) <{elem_type = i8, rawConstantIndices = array<i32: 64>}> : (!llvm.ptr) -> !llvm.ptr
    %13 = "llvm.ptrtoint"(%12) : (!llvm.ptr) -> i64
    %14 = llvm.add %13, %10 : i64
    %15 = "llvm.call"(%14) <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, callee = @malloc, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>}> : (i64) -> !llvm.ptr
    %16 = "llvm.ptrtoint"(%15) : (!llvm.ptr) -> i64
    %17 = llvm.sub %10, %11 : i64
    %18 = llvm.add %16, %17 : i64
    %19 = llvm.urem %18, %10 : i64
    %20 = llvm.sub %18, %19 : i64
    %21 = "llvm.inttoptr"(%20) : (i64) -> !llvm.ptr
    "llvm.br"(%9) [^1] : (i64) -> ()
  ^1(%22 : i64):
    %23 = llvm.icmp "slt" %22, %10 : i64
    "llvm.cond_br"(%23) [^2, ^3] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^2:
    %24 = "llvm.getelementptr"(%arg3, %22) <{elem_type = i32, rawConstantIndices = array<i32: -2147483648>}> : (!llvm.ptr, i64) -> !llvm.ptr
    %25 = "llvm.load"(%24) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
    %26 = llvm.sext %25 : i32 to i64
    %27 = llvm.mul %26, %6 : i64
    %28 = llvm.add %27, %8 : i64
    %29 = llvm.icmp "sge" %25, %1 : i32
    %30 = "llvm.select"(%29, %2, %3) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i64, i64) -> i64
    %31 = llvm.add %30, %28 : i64
    %32 = llvm.ashr %31, %7 : i64
    %33 = llvm.trunc %32 : i64 to i32
    %34 = "llvm.intr.smax"(%33, %4) : (i32, i32) -> i32
    %35 = "llvm.intr.smin"(%34, %5) : (i32, i32) -> i32
    %36 = llvm.trunc %35 : i32 to i8
    %37 = "llvm.getelementptr"(%21, %22) <{elem_type = i8, rawConstantIndices = array<i32: -2147483648>}> : (!llvm.ptr, i64) -> !llvm.ptr
    "llvm.store"(%36, %37) <{ordering = 0 : i64}> : (i8, !llvm.ptr) -> ()
    %38 = llvm.add %22, %11 : i64
    "llvm.br"(%38) [^1] : (i64) -> ()
  ^3:
    %39 = llvm.mul %10, %11 : i64
    %40 = "llvm.getelementptr"(%0) <{elem_type = i8, rawConstantIndices = array<i32: 1>}> : (!llvm.ptr) -> !llvm.ptr
    %41 = "llvm.ptrtoint"(%40) : (!llvm.ptr) -> i64
    %42 = llvm.mul %39, %41 : i64
    %43 = "llvm.getelementptr"(%arg8, %arg9) <{elem_type = i8, rawConstantIndices = array<i32: -2147483648>}> : (!llvm.ptr, i64) -> !llvm.ptr
    "llvm.intr.memcpy"(%43, %21, %42) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    "llvm.return"() : () -> ()
  }) {llvm.emit_c_interface} : () -> ()
  "llvm.func"() <{CConv = #llvm.cconv<ccc>, function_type = !llvm.func<void (!llvm.ptr, !llvm.ptr)>, linkage = #llvm.linkage<"external">, sym_name = "_mlir_ciface_rescale_down", visibility_ = 0 : i64}> ({
  ^4(%arg0 : !llvm.ptr, %arg1 : !llvm.ptr):
    %44 = "llvm.load"(%arg0) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<1 x i64>, !llvm.array<1 x i64>)>
    %45 = "llvm.extractvalue"(%44) <{position = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<1 x i64>, !llvm.array<1 x i64>)>) -> !llvm.ptr
    %46 = "llvm.extractvalue"(%44) <{position = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<1 x i64>, !llvm.array<1 x i64>)>) -> !llvm.ptr
    %47 = "llvm.extractvalue"(%44) <{position = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<1 x i64>, !llvm.array<1 x i64>)>) -> i64
    %48 = "llvm.extractvalue"(%44) <{position = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<1 x i64>, !llvm.array<1 x i64>)>) -> i64
    %49 = "llvm.extractvalue"(%44) <{position = array<i64: 4, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<1 x i64>, !llvm.array<1 x i64>)>) -> i64
    %50 = "llvm.load"(%arg1) <{ordering = 0 : i64}> : (!llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<1 x i64>, !llvm.array<1 x i64>)>
    %51 = "llvm.extractvalue"(%50) <{position = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<1 x i64>, !llvm.array<1 x i64>)>) -> !llvm.ptr
    %52 = "llvm.extractvalue"(%50) <{position = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<1 x i64>, !llvm.array<1 x i64>)>) -> !llvm.ptr
    %53 = "llvm.extractvalue"(%50) <{position = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<1 x i64>, !llvm.array<1 x i64>)>) -> i64
    %54 = "llvm.extractvalue"(%50) <{position = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<1 x i64>, !llvm.array<1 x i64>)>) -> i64
    %55 = "llvm.extractvalue"(%50) <{position = array<i64: 4, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i64, !llvm.array<1 x i64>, !llvm.array<1 x i64>)>) -> i64
    "llvm.call"(%45, %46, %47, %48, %49, %51, %52, %53, %54, %55) <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, callee = @rescale_down, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 10, 0>}> : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    "llvm.return"() : () -> ()
  }) {llvm.emit_c_interface} : () -> ()
}

