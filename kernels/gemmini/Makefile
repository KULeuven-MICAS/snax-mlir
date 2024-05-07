# Linking like this requires the binary to be compiled with -r for relocatable code
sp_tiled_matmul_ws.o: 
	compiler/snax-opt kernels/gemmini/sp_tiled_matmul_ws.mlir -p convert-acc-to-csr | mlir-opt-17 --test-lower-to-llvm | mlir-translate-17 --mlir-to-llvmir | clang-17 -x ir - -march=rv64g --target=riscv64-generic -nostdlib -r -o sp_tiled_matmul_ws.o

mlp4_32-baremetal:
	riscv64-unknown-elf-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -lm -lgcc -I/home/josse/phd/gemmini-rocc-tests/riscv-tests -I/home/josse/phd/gemmini-rocc-tests/riscv-tests/env -I/home/josse/phd/gemmini-rocc-tests -I/home/josse/phd/gemmini-rocc-tests/riscv-tests/benchmarks/common -DID_STRING=  -nostdlib -nostartfiles -static -T /home/josse/phd/gemmini-rocc-tests/riscv-tests/benchmarks/common/test.ld -DBAREMETAL=1  /home/josse/phd/gemmini-rocc-tests/mlps/mlp4_32.c  /home/josse/phd/snax-mlir/sp_tiled_matmul_ws.o  -o mlp4_32-baremetal /home/josse/phd/gemmini-rocc-tests/riscv-tests/benchmarks/common/syscalls.c /home/josse/phd/gemmini-rocc-tests/riscv-tests/benchmarks/common/crt.S
