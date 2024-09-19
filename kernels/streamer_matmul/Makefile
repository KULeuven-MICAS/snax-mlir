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
ACCFGOPT=accfg-config-overlap,
endif
ifdef ACCFG_BOTH
ACCFGOPT=accfg-dedup,accfg-config-overlap,
endif

SNAXOPTFLAGS = -p insert-accfg-op{accelerator=snax_gemm},convert-linalg-to-kernel,dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,reuse-memref-allocs,insert-sync-barrier,dispatch-regions,linalg-to-library-call,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space 
SNAXOPTSTREAMFLAGS = -p insert-accfg-op{accelerator=snax_gemm},convert-linalg-to-kernel,dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,${REMOVE_MEMREF_COPY}insert-sync-barrier,reuse-memref-allocs,test-add-mcycle-around-loop,snax-lower-mcycle,dispatch-regions,guarded-linalg-to-memref-stream,schedule-memref-linalg,stream-snaxify,convert-linalg-to-accfg,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space,function-constant-pinning,mlir-opt{executable=mlir-opt\ generic=true\ arguments="-cse,-canonicalize,-allow-unregistered-dialect,-mlir-print-op-generic"},${ACCFGOPT}convert-accfg-to-csr,

%.stream.snax-opt.mlir: %.preprocfinal.mlir
	$(SNAXOPT) $(SNAXOPTSTREAMFLAGS) -o $@ $<

ifndef SIZE_M
SIZE_M=16
endif
ifndef SIZE_N
SIZE_N=16
endif
ifndef SIZE_K
SIZE_K=16
endif

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
