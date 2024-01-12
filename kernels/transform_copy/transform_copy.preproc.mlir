builtin.module {
  func.func public @transform_copy(%arg0 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 0 : i32>, %arg1 : memref<8x8xi32, #tsl.tsl<[2, 4] -> (64, 4), [2, 4] -> (128, 16)>, 1 : i32>) {
    "memref.copy"(%arg0, %arg1) : (memref<8x8xi32, #tsl.tsl<[2, 4] -> (16, 4), [2, 4] -> (128, 32)>, 0 : i32>, memref<8x8xi32, #tsl.tsl<[2, 4] -> (64, 4), [2, 4] -> (128, 16)>, 1 : i32>) -> ()
    func.return
  }
}

