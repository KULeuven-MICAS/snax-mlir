# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ../../runtime/snax-gemmx.rules
include ../../runtime/Makefile.rules

TESTS =
TESTS += conv.x

MLIRPREPROCFLAGS = --linalg-generalize-named-ops
MLIRPREPROCFLAGS += --mlir-print-op-generic
MLIRPREPROCFLAGS += --mlir-print-local-scope

%.preprocfinal.mlir: %.mlir
	$(MLIROPT) $(MLIRPREPROCFLAGS) -o $@ $<


SNAXOPTFLAGS = --print-op-generic -p convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemmx},dispatch-kernels,convert-linalg-to-stream,fuse-streaming-regions,stream-bufferize#,snax-bufferize#,alloc-to-global#,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions{nb_cores=3},convert-stream-to-snax-stream,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space

CFLAGS += -std=gnu11
CFLAGS += -Wall -Wextra

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
