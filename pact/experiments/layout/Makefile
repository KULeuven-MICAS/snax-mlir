
TESTS += $(wildcard test_*)

generate:
	python3 generate_tests.py

run_%: %
	cd $< && make allrun

run: $(addprefix run_, $(TESTS))

traces_%: %
	cd $< && make traces

traces: $(addprefix traces_, $(TESTS))


parse_logs_%: %
	python3 parse_logs.py -i $</matmul.x.logs/trace_hart_00000000.trace.txt -o $</results.py
parse_logs:
	$(addprefix parse_logs_, $(TESTS))

all: $(addprefix run_, $(TESTS))

clean: 
	rm -rf test_*
