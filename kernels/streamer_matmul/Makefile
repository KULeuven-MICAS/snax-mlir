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

SNAXOPTFLAGS = -p insert-accfg-op{accelerator=snax_gemm},convert-linalg-to-kernel,dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,reuse-memref-allocs,insert-sync-barrier,dispatch-regions,linalg-to-library-call,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space
SNAXOPTSTREAMFLAGS = -p convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemm},dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions,guarded-linalg-to-memref-stream,schedule-memref-linalg,stream-snaxify,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space

%.stream.snax-opt.mlir: %.preprocfinal.mlir
	$(SNAXOPT) $(SNAXOPTSTREAMFLAGS) -o $@ $<



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
