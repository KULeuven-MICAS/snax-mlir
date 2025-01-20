module {
  func.func public @simple_copy(%arg0: memref<?xi32>, %arg1: memref<?xi32>) {
    "memref.copy"(%arg0, %arg1) : (memref<?xi32>, memref<?xi32>) -> ()
    return
  }
}
