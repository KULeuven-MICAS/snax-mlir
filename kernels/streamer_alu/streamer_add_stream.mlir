  func.func public @streamer_add(%arg0 : memref<16xi64>, %arg1 : memref<16xi64>, %arg2 : memref<16xi64>) {
  linalg.add ins(%arg0, %arg1 : memref<16xi64>, memref<16xi64>) outs(%arg2: memref<16xi64>)
  func.return
}
