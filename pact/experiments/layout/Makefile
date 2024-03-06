
TESTS += $(wildcard test_*)

generate:
	python3 generate_tests.py

run_%: %
	cd $< && make allrun

run: $(addprefix run_, $(TESTS))

traces_%: %
	cd $< && make traces

traces: $(addprefix traces_, $(TESTS))

parse_logs:
	python3 parse_logs.py

summary:
	python3 create_summary.py	

all: $(addprefix run_, $(TESTS))

clean: 
	rm -rf test_*
