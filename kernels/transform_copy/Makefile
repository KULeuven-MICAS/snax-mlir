# Courtesy of Federico Ficarelli

.DEFAULT_GOAL := all

include ../../runtime/snax-mac.rules
include ../../runtime/Makefile.rules

TESTS =
TESTS += transform_copy.x
TESTS += $(patsubst %_gen.preproc.mlir,%_gen.x,$(wildcard *_gen.preproc.mlir))

CFLAGS += -std=gnu11
CFLAGS += -Wall -Wextra

%.x: %.o %_main.o %_data.o
	$(LD) $(LDFLAGS) $^ -o $@

%_main.o: main.c
	$(PYTHON) gendata.py
	$(CC) $(CFLAGS) -I./$* -c $< -o $@

%_data.o: ./%/data.c
	$(CC) $(CFLAGS) -I./$* -c $< -o $@

 ./%/data.c:
	$(PYTHON) gendata.py

sim_%: %
	rm -fr ./logs/
	$(VLTSIM) $<

RUN = $(addprefix run_, $(TESTS))
$(RUN): run_%: sim_%
	mv logs $(subst sim_,,$<).logs

all: $(TESTS)

allrun: $(RUN)

printtests:
	echo $(TESTS)

printrun:
	echo $(RUN)

generate:
	$(PYTHON) gentests.py

clean:
	rm -fr *.ll12 *.x *.o *.logs/ logs/ data.h data.c *_gen.preproc.mlir *_gen *_gen_data.c transform_copy
