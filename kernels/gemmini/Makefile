.DEFAULT_GOAL := all

include ../../runtime/Makefile.rules

TESTS =
TESTS += tiled_matmul.acc-dialect.x

CFLAGS += -std=gnu11
CFLAGS += -Wall -Wextra


# Override snax-opt rules to avoid linalg-to-library-call pass
SNAXOPTACCFLAGS = -p insert-accfg-op{accelerator=gemmini},convert-linalg-to-accfg,convert-accfg-to-csr

tiled_matmul.mlir: tiled_matmul.transform.mlir
	# Use quick patch to remove transform module
	${MLIROPT} --pass-pipeline='builtin.module(transform-interpreter, test-transform-dialect-erase-schedule)' $^ | sed '/^  module @transforms attributes {transform.with_named_sequence} {$$/,/^  }$$/d' > $@


%.acc-dialect.snax-opt.mlir: %.preprocfinal.mlir
	$(SNAXOPT) $(SNAXOPTACCFLAGS) -o $@ $<

%.postproc.mlir: %.snax-opt.mlir
	cat $< | sed 's/arith.maximumf/arith.maxf/g' | sed 's/arith.minimumf/arith.minf/g' > $@

%.ll.mlir: %.postproc.mlir
	${MLIROPT} --test-lower-to-llvm -o $@ $<

%.ll: %.postproc.mlir
	${MLIRTRANSLATE} --mlir-to-llvmir -o $@ $<

%.x: %.ll
	${CC} -x ir $< -c -O3 -o $@ -Wno-override-module --target=riscv64-unknown-elf -mcpu=generic-rv64 -march=rv64gc


all: $(TESTS)

dump: $(TESTS)
	${OBJDUMP} -d $<


clean:
	rm -fr *.ll12 *.x *.o *.logs/ logs/ data.h data.c
