builtin.module {
  func.func public @transform_copy(%arg0 : memref<8x8xi32, #tsl.tsl<([1, 16] * [4, 2], [32, 128] * [4, 2])>, 0 : i32>, %arg1 : memref<8x8xi32, #tsl.tsl<([1, 64] * [4, 2], [16, 128] * [4, 2])>, 1 : i32>) {
    "memref.copy"(%arg0, %arg1) : (memref<8x8xi32, #tsl.tsl<([1, 16] * [4, 2], [32, 128] * [4, 2])>, 0 : i32>, memref<8x8xi32, #tsl.tsl<([1, 64] * [4, 2], [16, 128] * [4, 2])>, 1 : i32>) -> ()
    func.return
  }
}
