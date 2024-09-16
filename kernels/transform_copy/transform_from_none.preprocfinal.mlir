builtin.module {
  func.func public @transform_copy(%arg0 : memref<?x?xi32, "L3">, %arg1 : memref<?x?xi32, #tsl.tsl<[?, 4] -> (?, 4), [?, 4] -> (?, 1)>, "L1">) {
    "memref.copy"(%arg0, %arg1) : (memref<?x?xi32, "L3">, memref<?x?xi32, #tsl.tsl<[?, 4] -> (?, 4), [?, 4] -> (?, 1)>, "L1">) -> ()
    func.return
  }
}
