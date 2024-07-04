# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ../../runtime/snax-alu.rules
include ../../runtime/Makefile.rules

TESTS =
TESTS += untiled.acc_dialect.x
TESTS += tiled.acc_dialect.x
TESTS += tiled_pipelined.acc_dialect.x

CFLAGS += -std=gnu11
CFLAGS += -Wall -Wextra

MLIRPREPROCFLAGS += --allow-unregistered-dialect
MLIRPREPROC2FLAGS += --allow-unregistered-dialect
MLIRPREPROC3FLAGS += --allow-unregistered-dialect

ifdef NO_CHECK
 CFLAGS += -DNO_CHECK
endif


ifdef ACCFGOPT
 ACCFGOPTFLAGS=accfg-dedup,accfg-config-overlap,
endif
	

# Override snax-opt rules to avoid linalg-to-library-call pass
SNAXOPTACCFLAGS = -p insert-accfg-op{accelerator=snax_alu},convert-linalg-to-accfg,${ACCFGOPTFLAGS} #convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space,snax-lower-mcycle


# Set default 
ARRAY_SIZE ?= 2048
TILE_SIZE ?= 128

%.preprocfinal.sized.mlir: %.preprocfinal.mlir
	cat $< | sed -E 's/128/${ARRAY_SIZE}/gm;t' | sed -E 's/%tile_size = arith.constant 16 : index/%tile_size = arith.constant ${TILE_SIZE} : index/gm;t' > $@

%.acc_dialect.snax-opt.mlir: %.preprocfinal.sized.mlir
	$(SNAXOPT) $(SNAXOPTACCFLAGS) -o $@ $<

data.c data.h:
	$(PYTHON) gendata.py --array_size=${ARRAY_SIZE}

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
