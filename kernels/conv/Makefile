# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include /workspace/runtime/snax-gemmx.rules
include /workspace/runtime/Makefile.rules

TESTS =
TESTS += conv.x
TESTS += generated.x

MLIRPREPROCFLAGS = --linalg-generalize-named-ops
MLIRPREPROCFLAGS += --mlir-print-op-generic
MLIRPREPROCFLAGS += --mlir-print-local-scope

ifndef SCHEDULE_IDX
SCHEDULE_IDX=0
endif

%.preprocfinal.mlir: %.mlir
	$(MLIROPT) $(MLIRPREPROCFLAGS) -o $@ $<

SNAXOPTFLAGS = --print-op-generic -p convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemmx},dispatch-kernels,convert-linalg-to-stream,fuse-streaming-regions,stream-bufferize,snax-bufferize,alloc-to-global,autoflow-scheduler{schedule_idx=${SCHEDULE_IDX}},set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions{nb_cores=3},autoflow-layout-resolution,convert-linalg-to-accfg,test-add-mcycle-around-launch,convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,snax-lower-mcycle,clear-memory-space

CFLAGS += -std=gnu11
CFLAGS += -Wall -Wextra

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
	rm -fr *.ll12 *.x *.o *.logs/ logs/ data.h data.c *.mlir

# List of directories
CONV_DIRS := $(wildcard results/conv_*/)

# Variable to count completed jobs
COMPLETED := 0

# Default target
convs: $(CONV_DIRS)
	@echo "All jobs completed! Processed $(COMPLETED)/$(words $(CONV_DIRS)) directories."

# Target for each directory
$(CONV_DIRS):
	@echo "Processing $@... ($(COMPLETED)/$(words $(CONV_DIRS)))"
	@$(MAKE) -C $@ sim_generated.x > $@/sim_generated.log 2>&1
	@echo "sim_generated.x completed for $@" >> $@/sim_generated.log
	@$(MAKE) -C $@ traces >> $@/traces.log 2>&1
	@echo "traces completed for $@" >> $@/traces.log
	$(eval COMPLETED := $(shell echo $$(( $(COMPLETED) + 1 ))))

.PHONY: all $(CONV_DIRS)
