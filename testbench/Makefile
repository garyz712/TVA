# Makefile

# defaults
SIM ?= verilator
TOPLEVEL_LANG ?= verilog

EXTRA_ARGS += --trace --trace-structs

# VERILOG_SOURCES += $(PWD)/../src/attention_av_multiply.sv
# VERILOG_SOURCES += $(PWD)/../src/mul16.sv
# VERILOG_SOURCES += $(PWD)/../src/matmul_array.sv
# VERILOG_SOURCES += $(PWD)/../src/attention_score.sv
VERILOG_SOURCES += $(PWD)/../src/precision_assigner.sv
# use VHDL_SOURCES for VHDL files

# TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
TOPLEVEL = precision_assigner

# MODULE is the basename of the Python test file
MODULE = test_precision_assigner

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim
