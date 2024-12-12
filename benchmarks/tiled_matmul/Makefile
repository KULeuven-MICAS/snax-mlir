# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ../../runtime/snax-gemmx.rules
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

SNAXOPTFLAGS = -p convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemmx},dispatch-kernels,convert-linalg-to-stream,fuse-streaming-regions,snax-bufferize,alloc-to-global,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions{nb_cores=2},convert-stream-to-snax-stream,convert-linalg-to-accfg,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space,function-constant-pinning,mlir-opt{executable=mlir-opt\ generic=true\ arguments="-cse,-canonicalize,-allow-unregistered-dialect,-mlir-print-op-generic"},${ACCFGOPT}convert-accfg-to-csr,

CFLAGS += -std=gnu11
CFLAGS += -Wall -Wextra
# Needed for perfetto script
CFLAGS += -g
ifdef NO_CHECK
CFLAGS += -DNO_CHECK
endif

%.x: %.o main.o
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
