# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ../../runtime/snax-alu.rules
include ../../runtime/Makefile.rules

TESTS =
TESTS += streamer_mult.x

CFLAGS += -std=gnu11
CFLAGS += -Wall -Wextra

data.c data.h:
	$(PYTHON) gendata.py

%.x: main.o data.o
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