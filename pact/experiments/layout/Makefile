
TESTS += $(wildcard test_*)

generate:
	python3 generate_tests.py

run_%: %
	cd $< && make allrun

all: $(addprefix run_, $(TESTS))

clean: 
	rm -rf test_*
