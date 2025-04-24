# Simulator settings
SIM ?= icarus
TOPLEVEL_LANG ?= verilog

# Cocotb settings
export COCOTB_REDUCED_LOG_FMT=1
export PYTHONPATH := testbench:$(PYTHONPATH)

VERILOG_SOURCES = $(wildcard src/*.sv)

# Default target
all:
	@echo "Usage: make <module_name>"

# Dynamic target for module testing
%:
	$(MAKE) sim TOPLEVEL=$@ MODULE=test_$@

# Simulation target
sim:
	$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

# Clean target
clean:
	rm -rf sim_build results.xml *.vcd

.PHONY: all sim clean