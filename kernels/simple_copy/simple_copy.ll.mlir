module attributes {llvm.data_layout = ""} {
  llvm.func @simple_copy(%arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: !llvm.ptr<i32>, %arg6: !llvm.ptr<i32>, %arg7: i32, %arg8: i32, %arg9: i32) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>
    %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %11 = llvm.insertvalue %arg9, %10[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    llvm.call @snax_dma_1d_transfer(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (!llvm.ptr<i32>, !llvm.ptr<i32>, i32, i32, i32, !llvm.ptr<i32>, !llvm.ptr<i32>, i32, i32, i32) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_simple_copy(%arg0: !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>>, %arg1: !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>>) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %6 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %11 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    llvm.call @simple_copy(%1, %2, %3, %4, %5, %7, %8, %9, %10, %11) : (!llvm.ptr<i32>, !llvm.ptr<i32>, i32, i32, i32, !llvm.ptr<i32>, !llvm.ptr<i32>, i32, i32, i32) -> ()
    llvm.return
  }
  llvm.func @snax_dma_1d_transfer(%arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: !llvm.ptr<i32>, %arg6: !llvm.ptr<i32>, %arg7: i32, %arg8: i32, %arg9: i32) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %6 = llvm.mlir.constant(1 : index) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> : (i32) -> !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>>
    llvm.store %5, %7 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>
    %9 = llvm.insertvalue %arg5, %8[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %10 = llvm.insertvalue %arg6, %9[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %11 = llvm.insertvalue %arg7, %10[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %12 = llvm.insertvalue %arg8, %11[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %13 = llvm.insertvalue %arg9, %12[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> 
    %14 = llvm.mlir.constant(1 : index) : i32
    %15 = llvm.alloca %14 x !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)> : (i32) -> !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>>
    llvm.store %13, %15 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>>
    llvm.call @_mlir_ciface_snax_dma_1d_transfer(%7, %15) : (!llvm.ptr<struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>>, !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_snax_dma_1d_transfer(!llvm.ptr<struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>>, !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>>) attributes {llvm.emit_c_interface, sym_visibility = "private"}
}

