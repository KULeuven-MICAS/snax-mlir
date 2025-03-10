.DEFAULT_GOAL := all

include ../../runtime/Makefile.rules

TESTS =
TESTS += tiled_matmul.acc-dialect.x

CFLAGS += -std=gnu11
CFLAGS += -Wall -Wextra


# Override snax-opt rules to avoid linalg-to-library-call pass
SNAXOPTACCFLAGS = -p convert-linalg-to-gemmini-os,mlir-opt[cse,canonicalize],accfg-dedup,convert-gemmini-os-to-rocc
#SNAXOPTACCFLAGS = -p convert-linalg-to-gemmini-os

tiled_matmul.mlir: tiled_matmul.transform.mlir
	${MLIROPT} --pass-pipeline='builtin.module(test-transform-dialect-interpreter{bind-first-extra-to-ops=linalg.quantized_matmul}, test-transform-dialect-erase-schedule, linalg-generalize-named-ops)' $^ | sed -E 's/iterator_types =/library_call="gemmini_os", iterator_types =/gm;t' > $@

# don't hate me for this, but XDSL does not support anything named differently
# because I am a lazy man
# it's okay bcuz we're in a container, right? RIGHT????
/usr/bin/mlir-opt:
	bash -c '[ -e /usr/bin/mlir-opt ] || ln -s "$$(command -v mlir-opt-17)" /usr/bin/mlir-opt'

%.acc-dialect.snax-opt.mlir: %.preprocfinal.mlir /usr/bin/mlir-opt
	$(SNAXOPT) $(SNAXOPTACCFLAGS) -o $@ $<

%.postproc.mlir: %.snax-opt.mlir
	cat $< | sed 's/arith.maximumf/arith.maxf/g' | sed 's/arith.minimumf/arith.minf/g' > $@

%.ll.mlir: %.postproc.mlir
	${MLIROPT} --test-lower-to-llvm -o $@ $<

%.ll: %.postproc.mlir
	${MLIRTRANSLATE} --mlir-to-llvmir -o $@ $<

%.x: %.ll
	${CC} -x ir $< -c -O3 -o $@ -Wno-override-module --target=riscv64-unknown-elf -mcpu=generic-rv64 -march=rv64gc

allrun: all

all: $(TESTS)

dump: $(TESTS)
	${OBJDUMP} -d $<


clean:
	rm -fr *.ll12 *.x *.o *.logs/ logs/ data.h data.c
