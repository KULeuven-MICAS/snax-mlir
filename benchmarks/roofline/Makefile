# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ../../runtime/snax-gemmx.rules
include ../../runtime/Makefile.rules

TESTS += generated.x

MLIRPREPROCFLAGS = --linalg-generalize-named-ops
MLIRPREPROCFLAGS += --mlir-print-op-generic
MLIRPREPROCFLAGS += --mlir-print-local-scope

%.preprocfinal.mlir: %.mlir
	$(MLIROPT) $(MLIRPREPROCFLAGS) -o $@ $<

SNAXOPTFLAGS = -p convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemmx},dispatch-kernels,convert-linalg-to-stream,set-memory-space,set-memory-layout,realize-memref-casts,dce,insert-sync-barrier,dispatch-regions{nb_cores=3},convert-stream-to-snax-stream,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma{test_ignore_transform=true},memref-to-snax,snax-to-func,snax-lower-mcycle,clear-memory-space

GEN_DATA_OPTS += --m=${SIZE_M}
GEN_DATA_OPTS += --n=${SIZE_N}
GEN_DATA_OPTS += --k=${SIZE_K}

CFLAGS += -std=gnu11
CFLAGS += -Wall -Wextra
# Needed for perfetto script
CFLAGS += -g
ifdef NO_CHECK
CFLAGS += -DNO_CHECK
endif

data.c data.h:
	$(PYTHON) gendata.py ${GEN_DATA_OPTS}

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
	rm -fr *.ll12 *.x *.o *.logs/ logs/ data.h data.c *.png *.pdf *.vcd *.csv *.tar.gz
