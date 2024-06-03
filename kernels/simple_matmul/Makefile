# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ../../runtime/snax-gemm.rules
include ../../runtime/Makefile.rules

COMMA = ,
SPACE := $(null) #

TESTS =
TESTS += matmul.x
TESTS += matmul.acc_dialect.x

CFLAGS += -std=gnu11
CFLAGS += -Wall -Wextra

# Override snax-opt rules to avoid linalg-to-library-call pass
SNAX_PASSES = dispatch-kernels
SNAX_PASSES += set-memory-space
SNAX_PASSES += set-memory-layout
SNAX_PASSES += realize-memref-casts
SNAX_PASSES += insert-sync-barrier
SNAX_PASSES += dispatch-regions
SNAX_PASSES += insert-accfg-op{accelerator=snax_gemm}
SNAX_PASSES += snax-copy-to-dma
SNAX_PASSES += memref-to-snax
SNAX_PASSES += clear-memory-space
SNAX_PASSES += convert-linalg-to-accfg
SNAX_PASSES += mlir-opt[cse,canonicalize,loop-invariant-code-motion]
SNAX_PASSES += accfg-dedup
SNAX_PASSES += convert-accfg-to-csr
SNAX_PASSES += snax-to-func


SNAX_PASSES_UNIFIED = $(subst $(SPACE),$(COMMA),$(SNAX_PASSES))

%.acc_dialect.snax-opt.mlir: %.preprocfinal.mlir
	$(SNAXOPT) -p "$(SNAX_PASSES_UNIFIED)" $< -o $@

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
