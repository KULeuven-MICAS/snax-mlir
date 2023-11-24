builtin.module {
  func.func public @simple_alloc() -> memref<64xi32> {
    %0 = "memref.alloc"() {"alignment" = 64 : i64, operand_segment_sizes = array<i32: 0, 0>} : () -> memref<64xi32>
    func.return %0 : memref<64xi32>
  }
}


