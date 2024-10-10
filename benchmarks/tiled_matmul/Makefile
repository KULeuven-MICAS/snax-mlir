# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ../../runtime/snax-streamer-gemm.rules
include ../../runtime/Makefile.rules

TESTS += generated.x

# Don't remove memref copies if performing correctness checks
ifdef NO_CHECK 
REMOVE_MEMREF_COPY=test-remove-memref-copy,
else
REMOVE_MEMREF_COPY=
endif

# Allow these to be set from command line
ACCFGOPT=
ifdef DEDUP_ONLY
ACCFGOPT=accfg-dedup,
endif
ifdef OVERLAP_ONLY
ACCFGOPT=accfg-config-overlap,
endif
ifdef ACCFG_BOTH
ACCFGOPT=accfg-dedup,accfg-config-overlap,
endif

SNAXOPTFLAGS = -p insert-accfg-op{accelerator=snax_gemm},convert-linalg-to-kernel,dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,${REMOVE_MEMREF_COPY}insert-sync-barrier,reuse-memref-allocs,test-add-mcycle-around-loop,snax-lower-mcycle,dispatch-regions,convert-linalg-to-stream,convert-stream-to-snax-stream,convert-linalg-to-accfg,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space,function-constant-pinning,mlir-opt{executable=mlir-opt\ generic=true\ arguments="-cse,-canonicalize,-allow-unregistered-dialect,-mlir-print-op-generic"},${ACCFGOPT}convert-accfg-to-csr,


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
	rm -fr *.ll12 *.x *.o *.logs/ logs/ data.h data.c
