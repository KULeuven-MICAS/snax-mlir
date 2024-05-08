GEMMINI_ROCC_TESTS := /home/josse/phd/gemmini-rocc-tests
RISCV := /home/josse/phd/riscv-isa-sim/build
SNAX-OPT:= /repo/compiler/snax-opt
MLIR-OPT:= mlir-opt-17
MLIR-TRANS:= mlir-translate-17
CLANG := clang-17
GCC := riscv64-unknown-elf-gcc

.PHONY: patch spike

# Patch gemmini.h to not use the provided implementation and disable inlining
patch: ${GEMMINI_ROCC_TESTS}/include/gemmini.h
	git -c ${GEMMINI_ROCC_TESTS} apply gemmini.h.patch

# Linking like this requires the binary to be compiled with -r for relocatable code
sp_tiled_matmul_ws.o: sp_tiled_matmul_ws.mlir
	 ${SNAX-OPT} $< -p convert-acc-to-csr | ${MLIR-OPT} --test-lower-to-llvm | ${MLIR-TRANS} --mlir-to-llvmir | ${CLANG} -x ir - -march=rv64g --target=riscv64-generic -nostdlib -r -o $@

APP := mlp4_32-baremetal
SRCS := ${GEMMINI_ROCC_TESTS}/mlps/mlp4_32.c

${APP}: sp_tiled_matmul_ws.o
	${GCC} -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -lm -lgcc -I${GEMMINI_ROCC_TESTS}/riscv-tests -I${GEMMINI_ROCC_TESTS}/riscv-tests/env -I${GEMMINI_ROCC_TESTS} -I${GEMMINI_ROCC_TESTS}/benchmarks/common -DID_STRING=  -nostdlib -nostartfiles -static -T ${GEMMINI_ROCC_TESTS}/riscv-tests/benchmarks/common/test.ld -DBAREMETAL=1  ${SRCS}  $<  -o mlp4_32-baremetal ${GEMMINI_ROCC_TESTS}/riscv-tests/benchmarks/common/syscalls.c ${GEMMINI_ROCC_TESTS}/riscv-tests/benchmarks/common/crt.S

spike: mlp4_32-baremetal
	${RISCV}/spike --extension=gemmini $<
