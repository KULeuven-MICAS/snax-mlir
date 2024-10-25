# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ./snax-gemmx-xdma.rules
# include ../../runtime/snax-gemmx.rules
#
include ../../runtime/Makefile.rules

TESTS =
TESTS += toy.x

MLIRPREPROCFLAGS = --linalg-generalize-named-ops
MLIRPREPROCFLAGS += --mlir-print-op-generic
MLIRPREPROCFLAGS += --mlir-print-local-scope

%.preprocfinal.mlir: %.mlir
	$(MLIROPT) $(MLIRPREPROCFLAGS) -o $@ $< --allow-unregistered-dialect

# snitch only:
# SNAXOPTFLAGS = -p convert-linalg-to-kernel,dispatch-kernels,convert-linalg-to-stream,fuse-streaming-regions,stream-bufferize,snax-bufferize,alloc-to-global,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions{nb_cores=2},convert-stream-to-snax-stream,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space,convert-kernel-to-linalg,test-rescale-to-trunc

# addition of gemmx:
# SNAXOPTFLAGS = -p convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemmx},dispatch-kernels,convert-linalg-to-stream,fuse-streaming-regions,stream-bufferize,snax-bufferize,alloc-to-global,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions{nb_cores=2},convert-stream-to-snax-stream,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma{test_ignore_transform=true},memref-to-snax,snax-to-func,clear-memory-space,convert-kernel-to-linalg,test-rescale-to-trunc

# addition of xdma:
# SNAXOPTFLAGS = -p convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemmx},insert-accfg-op{accelerator=snax_xdma},dispatch-kernels,convert-linalg-to-stream,fuse-streaming-regions,stream-bufferize,snax-bufferize,alloc-to-global,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions{nb_cores=2},convert-stream-to-snax-stream,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma{test_ignore_transform=true},memref-to-snax,snax-to-func,clear-memory-space,convert-kernel-to-linalg,test-rescale-to-trunc,linalg-to-library-call

# with all optimizations:
SNAXOPTFLAGS = -p 'convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemmx},insert-accfg-op{accelerator=snax_xdma},dispatch-kernels,convert-linalg-to-stream,fuse-streaming-regions,stream-bufferize,snax-bufferize,alloc-to-local,set-memory-space,realize-memref-casts,set-memory-layout,insert-sync-barrier,dispatch-regions{nb_cores=2},convert-stream-to-snax-stream,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma{test_ignore_transform=true},memref-to-snax,clear-memory-space,mlir-opt{generic=true arguments=-cse,-canonicalize,-allow-unregistered-dialect,-mlir-print-op-generic},snax-to-func{test_static_alloc=true},convert-kernel-to-linalg,test-rescale-to-trunc,linalg-to-library-call'


%.snax-opt.mlir: %.preprocfinal.mlir
	$(SNAXOPT) --allow-unregistered-dialect $(SNAXOPTFLAGS) -o $@ $<


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
