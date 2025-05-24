import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import random

DATA_WIDTH = 16
NUM_TOKENS = 196
E = 128

def flatten_2d_array(array, width=DATA_WIDTH):
    """Flatten a 2D array (NUM_TOKENS x E) into 1D binary value."""
    flat = 0
    for t in reversed(range(NUM_TOKENS)):
        for d in reversed(range(E)):
            flat = (flat << width) | array[t][d]
    return flat

def compute_addr(t, d):
    return t * E + d

@cocotb.test()
async def test_positional_encoding(dut):
    """Test full positional encoding logic including ROM add."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Load ROM from file
    rom_data = {}
    with open("pos_embed_fp16.mem", "r") as f:
        for i, line in enumerate(f):
            rom_data[i] = int(line.strip(), 16)

    # Initialize input embedding: A_in[t][e] = t+e
    A_array = [[(t + d) & 0xFFFF for d in range(E)] for t in range(NUM_TOKENS)]
    A_in_val = flatten_2d_array(A_array)
    dut.A_in.value = A_in_val

    # Start FSM
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done signal
    while not dut.done.value:
        await RisingEdge(dut.clk)

    assert dut.out_valid.value == 1, "Output not marked as valid!"

    # Check outputs
    out_flat = dut.out_embed.value.integer
    for t in range(NUM_TOKENS):
        for d in range(E):
            idx = t * E + d
            expected = (A_array[t][d] + rom_data[idx]) & 0xFFFF
            shift = idx * DATA_WIDTH
            mask = (1 << DATA_WIDTH) - 1
            actual = (out_flat >> shift) & mask
            assert actual == expected, f"out_embed[{t},{d}] = {actual} != {expected}"
