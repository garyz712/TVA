# Makefile

# defaults
SIM ?= verilator
TOPLEVEL_LANG ?= verilog

# EXTRA_ARGS += --trace --trace-structs

VERILOG_SOURCES += $(PWD)/../src/*.sv
# use VHDL_SOURCES for VHDL files

# TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
TOPLEVEL = self_attention_top

# MODULE is the basename of the Python test file
MODULE = test_self_attention_top
# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim
