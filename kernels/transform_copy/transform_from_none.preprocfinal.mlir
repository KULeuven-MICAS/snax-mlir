builtin.module {
  func.func public @transform_copy(%arg0 : memref<?x?xi32, 0 : i32>, %arg1 : memref<?x?xi32, #tsl.tsl<[?, 4] -> (?, 16), [?, 4] -> (?, 4)>, 1 : i32>) {
    "memref.copy"(%arg0, %arg1) : (memref<?x?xi32, 0 : i32>, memref<?x?xi32, #tsl.tsl<[?, 4] -> (?, 16), [?, 4] -> (?, 4)>, 1 : i32>) -> ()
    func.return
  }
}
