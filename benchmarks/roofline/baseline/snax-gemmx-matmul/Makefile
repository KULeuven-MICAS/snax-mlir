# Copyright 2023 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

APP     = snax-gemmx-matmul

INCDIRS = data

INCDIRS += ../../snax/gemmx/include

# Include this binary in the final build
# RISCV_LDFLAGS += ../../snax/gemmx/build/snax-gemmx-lib.o

CFLAGS += -I$(SNITCH_SW_PATH)/target/snitch_cluster/sw/snax/gemm/include
LDFLAGS += $(SNITCH_SW_PATH)/target/snitch_cluster/sw/snax/gemm/build/snax-gemm-lib.o

SRCS    = src/snax-gemmx-matmul.c

include ../common.mk
include ./data/Makefile

$(DEP): $(DATA_H)



# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ../../../../runtime/snax-gemmx.rules
include ../../../../runtime/Makefile.rules

TESTS = matmul.x
TESTS += matmul.x
TESTS += transform_matmul.x

CFLAGS += -std=gnu11
CFLAGS += -Wall -Wextra

data.c data.h:
	$(PYTHON) gendata.py

matmul.x: main.o data.o
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
