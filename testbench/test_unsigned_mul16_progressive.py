import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotb.clock import Clock
import random

@cocotb.test()
async def test_unsigned_mul16_progressive(dut):
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")  # 100 MHz clock
    cocotb.start_soon(clock.start())

    # Apply reset
    dut.rst_n.value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Test INT4 multiplication
    a4 = random.randint(0, 15)  # 4-bit random value
    b4 = random.randint(0, 15)  # 4-bit random value
    a = a4 & 0xF  # Zero-extend to 16 bits
    b = b4 & 0xF  # Zero-extend to 16 bits
    dut.a.value = a
    dut.b.value = b
    await RisingEdge(dut.clk)
    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    await RisingEdge(dut.clk)  # 1 cycle delay for p4
    await RisingEdge(dut.clk)  # 1 cycle delay for p4
    assert dut.out4_valid.value == 1, "out4_valid should be high for INT4 test"
    p4_expected = (a4 * b4) & 0xFF  # Expected 8-bit product
    assert dut.p4.value == p4_expected, f"INT4 product mismatch: {dut.p4.value} != {p4_expected}"

    # Test INT8 multiplication
    a8 = random.randint(0, 255)  # 8-bit random value
    b8 = random.randint(0, 255)  # 8-bit random value
    a = a8 & 0xFF  # Zero-extend to 16 bits
    b = b8 & 0xFF  # Zero-extend to 16 bits
    dut.a.value = a
    dut.b.value = b
    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)  # 2 cycle delay for p8
    await RisingEdge(dut.clk)  # 2 cycle delay for p8
    assert dut.out8_valid.value == 1, "out8_valid should be high for INT8 test"
    p8_expected = (a8 * b8) & 0xFFFF  # Expected 16-bit product
    assert dut.p8.value == p8_expected, f"INT8 product mismatch: {dut.p8.value} != {p8_expected}"

    # Test full 16-bit multiplication
    a16 = random.randint(0, 65535)  # 16-bit random value
    b16 = random.randint(0, 65535)  # 16-bit random value
    dut.a.value = a16
    dut.b.value = b16
    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    for _ in range(5):  # 4 cycle delay for p16
        await RisingEdge(dut.clk)
    assert dut.out16_valid.value == 1, "out16_valid should be high for 16-bit test"
    p16_expected = (a16 * b16) & 0xFFFFFFFF  # Expected 32-bit product
    assert dut.p16.value == p16_expected, f"16-bit product mismatch: {dut.p16.value} != {p16_expected}"
