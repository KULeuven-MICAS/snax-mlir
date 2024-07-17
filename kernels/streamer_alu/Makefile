# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ../../runtime/snax-alu.rules
include ../../runtime/Makefile.rules

TESTS =
TESTS += streamer_add.x
TESTS += streamer_add_stream.x

CFLAGS += -std=gnu11
CFLAGS += -Wall -Wextra

SNAXOPTFLAGS = -p dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions,insert-accfg-op{accelerator=snax_alu},convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space


# override for streamer_add_stream.x, without insert-accfg-op (this is only temporary)
SNAXOPTFLAGS2 = -p dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space
streamer_add_stream.snax-opt.mlir: streamer_add_stream.preprocfinal.mlir
	$(SNAXOPT) $(SNAXOPTFLAGS2) -o $@ $<


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
