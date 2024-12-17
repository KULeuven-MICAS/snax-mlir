# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ../../runtime/snax-gemmx.rules
include ../../runtime/Makefile.rules

TESTS =
TESTS += quantized_matmul.x
TESTS += tiled_quantized_matmul.x

SNAXOPTFLAGS = -p test-transform-dialect-erase-schedule,convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemmx},dispatch-kernels,convert-linalg-to-stream,fuse-streaming-regions,snax-bufferize,alloc-to-global,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions{nb_cores=2},convert-stream-to-snax-stream,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space

CFLAGS += -std=gnu11
CFLAGS += -Wall -Wextra

quantized_matmul.mlir: quantized_matmul.py
	$(PYTHON) quantized_matmul.py

tiled_quantized_matmul.transform.mlir: tiled_quantized_matmul.py
	$(PYTHON) tiled_quantized_matmul.py

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
	rm -fr *.mlir *.ll12 *.x *.o *.logs/ logs/ data.h data.c
