builtin.module {
  func.func public @transform_copy(%arg0 : memref<8x8xi32, #tsl.tsl<([1, 4] * [4, 2], [8, 32] * [4, 2])>, 0 : i32>, %arg1 : memref<8x8xi32, #tsl.tsl<([1, 16] * [4, 2], [4, 32] * [4, 2])>, 1 : i32>) {
    "memref.copy"(%arg0, %arg1) : (memref<8x8xi32, #tsl.tsl<([1, 4] * [4, 2], [8, 32] * [4, 2])>, 0 : i32>, memref<8x8xi32, #tsl.tsl<([1, 16] * [4, 2], [4, 32] * [4, 2])>, 1 : i32>) -> ()
    func.return
  }
}
