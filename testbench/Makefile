# Makefile

# defaults
SIM ?= verilator
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES += $(PWD)/../src/matmul_array.sv
# use VHDL_SOURCES for VHDL files

# TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
TOPLEVEL = matmul_array

# MODULE is the basename of the Python test file
MODULE = test_matmul_array

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim
