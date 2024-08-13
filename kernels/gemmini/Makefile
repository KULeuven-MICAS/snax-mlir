.DEFAULT_GOAL := all

include ../../runtime/Makefile.rules

TESTS =
TESTS += tiled_matmul.acc-dialect.x

CFLAGS += -std=gnu11
CFLAGS += -Wall -Wextra


# Override snax-opt rules to avoid linalg-to-library-call pass
SNAXOPTACCFLAGS = -p insert-accfg-op{accelerator=gemmini},convert-linalg-to-accfg #,convert-accfg-to-csr

tiled_matmul.mlir: tiled_matmul.transform.mlir
	# Use quick patch to remove transform module
	${MLIROPT} --pass-pipeline='builtin.module(transform-interpreter, test-transform-dialect-erase-schedule)' $^ | sed '/^  module @transforms attributes {transform.with_named_sequence} {$$/,/^  }$$/d' > $@


%.acc-dialect.snax-opt.mlir: %.preprocfinal.mlir
	echo lol
	$(SNAXOPT) $(SNAXOPTACCFLAGS) -o $@ $<

data.c data.h:
	$(PYTHON) gendata.py

%.x: %.o main.o data.o
	$(LD) $(LDFLAGS) $^ -o $@

sim_%: %
	rm -fr ./logs/
	$(VLTSIM) $<

RUN = $(addprefix run_, $(TESTS))
$(RUN): run_%: sim_%
	mv logs $(subst sim_,,$<).logs

all: $(TESTS)

allrun: $(RUN)

clean:
	rm -fr *.ll12 *.x *.o *.logs/ logs/ data.h data.c
