import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotb.clock import Clock
import random

NUM_TOKENS = 196
E = 128
DATA_WIDTH = 16

def compute_addr(token_idx, dim):
    return token_idx * E + dim

@cocotb.test()
async def test_rom_basic_read(dut):
    """Test that ROM returns correct values for known addresses."""

    # Start a 10ns period clock on clk
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # Apply reset
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1

    # Wait a few cycles after reset
    for _ in range(2):
        await RisingEdge(dut.clk)

    # Load the memory file and read it for checking values
    rom_data = {}
    with open("pos_embed_fp16.mem", "r") as f:
        for i, line in enumerate(f):
            val = int(line.strip(), 16)
            rom_data[i] = val

    # Test random addresses
    for _ in range(50):
        token_idx = random.randint(0, NUM_TOKENS - 1)
        dim = random.randint(0, E - 1)
        addr = compute_addr(token_idx, dim)

        dut.token_idx.value = token_idx
        dut.dim.value = dim
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")  # Wait for pos_val to update

        expected = rom_data.get(addr, None)
        if expected is None:
            raise ValueError(f"Address {addr} not found in ROM data.")

        # print (dut.pos_val.value)
        assert dut.pos_val.value == expected, \
            f"ROM[{addr}] = {int(dut.pos_val.value)} != {expected}"

    # Test boundary values
    for token_idx, dim in [(0, 0), (NUM_TOKENS - 1, E - 1)]:
        addr = compute_addr(token_idx, dim)
        dut.token_idx.value = token_idx
        dut.dim.value = dim
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")
        assert dut.pos_val.value == rom_data[addr], \
            f"ROM[{addr}] = {int(dut.pos_val.value)} != {rom_data[addr]}"
