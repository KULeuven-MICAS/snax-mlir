func.func public @simple_alloc() -> memref<10xi32, 1 : i32> {
	%alloc = memref.alloc() {alignment = 256 : i64} : memref<10xi32, 1 : i32>
	return %alloc : memref<10xi32, 1 : i32>
}
