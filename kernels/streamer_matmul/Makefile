# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ../../runtime/snax-kul-narrow.rules
include ../../runtime/Makefile.rules

TESTS =
# TESTS += matmul.x
# TESTS += half_tiled_matmul.x
# TESTS += transform_matmul.x
# TESTS += dynamic_matmul.x
TESTS += matmul_stream.x

# these passes must still be automized:
# dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions
SNAXOPTFLAGS = -p insert-accfg-op{accelerator=snax_gemmx},dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions,guarded-linalg-to-memref-stream,schedule-memref-linalg,stream-snaxify,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space

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
