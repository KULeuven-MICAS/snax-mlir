# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ../../runtime/snax-streamer-gemm.rules
include ../../runtime/Makefile.rules

# For static shapes, the stream lowering can be used with the suffix .stream

TESTS =
TESTS += matmul.x
TESTS += half_tiled_matmul.x
TESTS += transform_matmul.stream.x
TESTS += dynamic_matmul.x
TESTS += matmul_stream.stream.x

# only append this variable if it is generated
ifneq ("$(wildcard generated.transform.mlir)", "")
TESTS += generated.stream.x
endif

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
ACCFGOPT=accfg-config-overlap
endif
ifdef ACCFG_BOTH
ACCFGOPT=accfg-dedup,accfg-config-overlap
endif

SNAXOPTSTREAMFLAGS = -p insert-accfg-op{accelerator=snax_gemm},dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,${REMOVE_MEMREF_COPY}insert-sync-barrier,reuse-memref-allocs,dispatch-regions,guarded-linalg-to-memref-stream,schedule-memref-linalg,stream-snaxify,convert-linalg-to-accfg,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space,function-constant-pinning,mlir-opt{executable=mlir-opt-17\ generic=true\ arguments="-cse,-canonicalize,-allow-unregistered-dialect,-mlir-print-op-generic"},${ACCFGOPT}convert-accfg-to-csr

%.stream.snax-opt.mlir: %.preprocfinal.mlir
	$(SNAXOPT) $(SNAXOPTSTREAMFLAGS) -o $@ $<



CFLAGS += -std=gnu11
CFLAGS += -Wall -Wextra
# Needed for perfetto script
CFLAGS += -g
ifdef NO_CHECK
CFLAGS += -DNO_CHECK
endif

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
