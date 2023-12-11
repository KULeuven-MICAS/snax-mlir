func.func public @simple_alloc() -> (memref<10xi32, 1 : i32>) {
  %alloc = "memref.alloc"() {"alignment" = 64 : i64, operand_segment_sizes = array<i32: 0, 0>} : () -> memref<10xi32, 1 : i32>
  return %alloc : memref<10xi32, 1 : i32>
}
