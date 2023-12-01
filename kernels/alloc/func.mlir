func.func public @simple_alloc() -> (memref<64xi32>) {
  %alloc = "memref.alloc"() {"alignment" = 64 : i64, operand_segment_sizes = array<i32: 0, 0>} : () -> memref<64xi32>
  return %alloc : memref<64xi32>
}
