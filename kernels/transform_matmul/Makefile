# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ../../runtime/snax-gemm.rules
include ../../runtime/Makefile.rules

TESTS =
TESTS += matmul.x
TESTS += matmul.acc_dialect.x

CFLAGS += -std=gnu11
CFLAGS += -Wall -Wextra

# Override snax-opt rules to avoid linalg-to-library-call pass
SNAXOPTACCFLAGS = -p dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions,insert-accfg-op{accelerator=snax_gemm},convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space

%.acc_dialect.snax-opt.mlir: %.preprocfinal.mlir
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
