module {
  func.func public @streamer_add(%arg0: memref<16xf64>, %arg1: memref<16xf64>, %arg2: memref<16xf64>) {
    linalg.add ins(%arg0, %arg1 : memref<16xf64>, memref<16xf64>) outs(%arg2 : memref<16xf64>)
    return
  }
}
