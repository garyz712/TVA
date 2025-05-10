import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import random
import math

# Helper function to convert floating-point to Q1.N fixed-point (two's complement)
def float_to_q1_n(value, n_frac_bits):
    # Scale by 2^n_frac_bits, round, and clamp to 16-bit two's complement
    scaled = round(value * (1 << n_frac_bits))
    return scaled & 0xFFFF if scaled >= 0 else (scaled + (1 << 16)) & 0xFFFF

# Helper function to convert Q1.N fixed-point back to float
def q1_n_to_float(value, n_frac_bits):
    # Sign-extend if negative
    if value & 0x8000:
        value = value - (1 << 16)
    return value / (1 << n_frac_bits)

# Helper function to compute expected product in Q1.N format
def expected_product(a, b, in_frac_bits, out_frac_bits, out_bits):
    # Convert inputs to float
    a_float = q1_n_to_float(a, in_frac_bits)
    b_float = q1_n_to_float(b, in_frac_bits)
    # Compute product
    product = a_float * b_float
    # Convert to output Q-format
    scaled = round(product * (1 << out_frac_bits))
    # Extract relevant bits
    shift = out_frac_bits + 1 - out_bits  # Adjust for output bit-width
    if shift > 0:
        scaled = scaled >> shift
    mask = (1 << out_bits) - 1
    return scaled & mask

@cocotb.test()
async def test_four_stage_multiplier(dut):
    """Test the four-stage multiplier for Q1.2, Q1.6, Q1.14 outputs with positive and negative inputs"""
    # Initialize clock (10ns period)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset the DUT
    dut.rst_n.value = 0
    dut.a.value = 0
    dut.b.value = 0
    dut.valid_in.value = 0
    await Timer(20, units="ns")
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Test configurations for Q1.2, Q1.6, Q1.14
    test_cases = [
        # Q1.2 inputs (2 fractional bits)
        {"in_format": 2, "test_values": [
            (0.5, 0.5), (0.75, -0.25), (-0.5, 0.5), (-0.75, -0.75), (0.0, 1.0), (1.0, -1.0)
        ]},
        # Q1.6 inputs (6 fractional bits)
        {"in_format": 6, "test_values": [
            (0.5, 0.5), (0.875, -0.125), (-0.5, 0.25), (-0.625, -0.625), (0.0, 0.5), (0.5, -0.5)
        ]},
        # Q1.14 inputs (14 fractional bits)
        {"in_format": 14, "test_values": [
            (0.5, 0.5), (0.999, -0.001), (-0.5, 0.1), (-0.9, -0.9), (0.0, 0.01), (0.01, -0.01)
        ]}
    ]

    for test_case in test_cases:
        in_frac_bits = test_case["in_format"]
        for a_float, b_float in test_case["test_values"]:
            # Convert inputs to Q1.N format
            a = float_to_q1_n(a_float, in_frac_bits)
            b = float_to_q1_n(b_float, in_frac_bits)

            # Drive inputs
            dut.a.value = a
            dut.b.value = b
            dut.valid_in.value = 1
            await RisingEdge(dut.clk)

            # Wait for Q1.2 output (1 cycle latency)
            await RisingEdge(dut.clk)
            if dut.q1_2_valid.value:
                expected = expected_product(a, b, in_frac_bits, 2, 4)
                actual = int(dut.q1_2_out.value)
                assert actual == expected, f"Q1.2 failed: a={a_float}, b={b_float}, expected={expected}, got={actual}"

            # Wait for Q1.6 output (1 more cycle, total 2 cycles)
            await RisingEdge(dut.clk)
            if dut.q1_6_valid.value:
                expected = expected_product(a, b, in_frac_bits, 6, 8)
                actual = int(dut.q1_6_out.value)
                assert actual == expected, f"Q1.6 failed: a={a_float}, b={b_float}, expected={expected}, got={actual}"

            # Wait for Q1.14 output (2 more cycles, total 4 cycles)
            await RisingEdge(dut.clk)
            await RisingEdge(dut.clk)
            if dut.q1_14_valid.value:
                expected = expected_product(a, b, in_frac_bits, 14, 16)
                actual = int(dut.q1_14_out.value)
                assert actual == expected, f"Q1.14 failed: a={a_float}, b={b_float}, expected={expected}, got={actual}"

            # Clear valid_in
            dut.valid_in.value = 0
            await RisingEdge(dut.clk)

    # Run a few extra cycles to ensure pipeline clears
    for _ in range(5):
        await RisingEdge(dut.clk)

    # Final checks: ensure outputs are invalid after pipeline flush
    assert dut.q1_2_valid.value == 0, "Q1.2 valid should be 0 after flush"
    assert dut.q1_6_valid.value == 0, "Q1.6 valid should be 0 after flush"
    assert dut.q1_14_valid.value == 0, "Q1.14 valid should be 0 after flush"