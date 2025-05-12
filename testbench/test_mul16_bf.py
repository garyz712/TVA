
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import random

@cocotb.test()
async def test_mul16_progressive(dut):
    # Initialize clock
    clock = Clock(dut.clk, 10, units="ns")  # 100 MHz clock
    cocotb.start_soon(clock.start())

    # Reset the DUT
    dut.rst_n.value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await Timer(20, units="ns")

    # Test case: Multiply -7 (Q0.15: 0xFFF9) and -14 (Q0.15: 0xFFF2)
    a = 0xFFF9  # -7 in Q0.15
    b = 0xFFF2  # -14 in Q0.15
    expected_q1_6 = 0xF2  # Approximate Q1.6 result of 98 (from image)
    expected_q1_14 = 0x0062  # Q1.14 result of 98
    expected_q1_30 = 0x00000062  # Q1.30 result of 98

    dut.a.value = a
    dut.b.value = b
    dut.in_valid.value = 1

    # Wait for 1 cycle (Q1.6)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    assert dut.q1_6_valid.value == 1, "Q1.6 valid should be high"
    assert dut.q1_6_out.value == expected_q1_6, f"Q1.6 output mismatch: got {hex(dut.q1_6_out.value)}, expected {hex(expected_q1_6)}"

    # Wait for 2 cycles total (Q1.14)
    await RisingEdge(dut.clk)
    assert dut.q1_14_valid.value == 1, "Q1.14 valid should be high"
    assert dut.q1_14_out.value == expected_q1_14, f"Q1.14 output mismatch: got {hex(dut.q1_14_out.value)}, expected {hex(expected_q1_14)}"

    # Wait for 4 cycles total (Q1.30)
    await RisingEdge(dut.clk)
    assert dut.q1_30_valid.value == 1, "Q1.30 valid should be high"
    assert dut.q1_30_out.value == expected_q1_30, f"Q1.30 output mismatch: got {hex(dut.q1_30_out.value)}, expected {hex(expected_q1_30)}"

    dut.in_valid.value = 0
    await Timer(20, units="ns")

    # Additional random tests
    for _ in range(5):
        a = random.randint(-32768, 32767)  # Q0.15 range
        b = random.randint(-32768, 32767)
        dut.a.value = a & 0xFFFF
        dut.b.value = b & 0xFFFF
        dut.in_valid.value = 1
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        product = (a * b) & 0xFFFFFFFF  # Full 32-bit product
        assert dut.q1_30_out.value == product, f"Random test failed: got {hex(dut.q1_30_out.value)}, expected {hex(product)}"
        dut.in_valid.value = 0
        await Timer(20, units="ns")
