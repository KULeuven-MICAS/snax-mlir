# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ../../runtime/snax-gemmx.rules
include ../../runtime/Makefile.rules

TESTS += generated.x

SNAXOPTFLAGS = -p convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemmx},dispatch-kernels,set-memory-space,set-memory-layout{gemm_layout=${LAYOUT}},realize-memref-casts,insert-sync-barrier,dispatch-regions{nb_cores=3},guarded-linalg-to-memref-stream,schedule-memref-linalg,stream-snaxify,convert-linalg-to-accfg,test-add-mcycle-around-launch,convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,snax-lower-mcycle,clear-memory-space

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
	$(VLTSIM) $< --vcd

sim.csv: sim.vcd
	vcd-to-csv

plots: sim.csv
	$(PYTHON) ../../util/visualization/gemmx.py --input_file sim.csv --output_path ./

RUN = $(addprefix run_, $(TESTS))
$(RUN): run_%: sim_%
	mv logs $(subst sim_,,$<).logs

all: $(TESTS)

allrun: $(RUN)

clean:
	rm -fr *.ll12 *.x *.o *.logs/ logs/ data.h data.c *.png *.pdf *.vcd *.csv *.tar.gz
