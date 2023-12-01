# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ../../runtime/Makefile.rules

TESTS =
TESTS += func.x

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
	rm -fr *.ll12 *.x *.o *.logs/ logs/
