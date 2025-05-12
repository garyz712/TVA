import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# Convert float to signed Q0.N fixed-point (two's complement)
def float_to_q0_n(value, n_frac_bits):
    if not (-1 <= value < 1):
        raise ValueError(f"Q0.{n_frac_bits} format supports [-1.0, 1.0) range")
    scaled = round(value * (1 << n_frac_bits))
    total_bits = n_frac_bits + 1
    if scaled < 0:
        scaled = (1 << total_bits) + scaled
    return scaled & ((1 << total_bits) - 1)

# Convert Q0.N fixed-point (two's complement) to float
def q0_n_to_float(value, n_frac_bits):
    total_bits = n_frac_bits + 1
    if value & (1 << (total_bits - 1)):
        value -= (1 << total_bits)
    return value / (1 << n_frac_bits)

# Compute expected product in Q0.N format
def expected_product_q0_signed(a, b, in_frac_bits, out_frac_bits, out_bits):
    a_float = q0_n_to_float(a, in_frac_bits)
    b_float = q0_n_to_float(b, in_frac_bits)
    product = a_float * b_float
    scaled = round(product * (1 << out_frac_bits))
    shift = out_frac_bits + 1 - out_bits
    if shift > 0:
        scaled = scaled >> shift
    if scaled < 0:
        scaled = (1 << out_bits) + scaled
    return scaled & ((1 << out_bits) - 1)

@cocotb.test()
async def test_four_stage_multiplier_q0_signed(dut):
    """Test the four-stage multiplier for Q0.3, Q0.7, Q0.15 outputs (signed fixed-point)"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset DUT
    dut.rst_n.value = 0
    dut.a.value = 0
    dut.b.value = 0
    dut.valid_in.value = 0
    await Timer(20, units="ns")
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Define test cases for Q0.3 (4-bit), Q0.7 (8-bit), Q0.15 (16-bit)
    test_cases = [
        {"in_format": 3, "test_values": [
            (0.0, 0.0), (0.875, -0.875), (-0.5, 0.5), (-0.875, -0.875)
        ]},
        {"in_format": 7, "test_values": [
            (0.0, 0.0), (0.99, -0.125), (-0.5, 0.25), (-0.75, -0.75)
        ]},
        {"in_format": 15, "test_values": [
            (0.0, 0.0), (0.999, -0.001), (-0.5, 0.1), (-0.999, -0.999)
        ]}
    ]

    for test_case in test_cases:
        in_frac_bits = test_case["in_format"]
        for a_float, b_float in test_case["test_values"]:
            a = float_to_q0_n(a_float, in_frac_bits)
            b = float_to_q0_n(b_float, in_frac_bits)

            dut.a.value = a
            dut.b.value = b
            dut.valid_in.value = 1
            await RisingEdge(dut.clk)

            # Q0.3 (4-bit) output: after 1 cycle
            await RisingEdge(dut.clk)
            if hasattr(dut, "q0_3_valid") and dut.q0_3_valid.value:
                expected = expected_product_q0_signed(a, b, in_frac_bits, 3, 4)
                actual = int(dut.q0_3_out.value)
                assert actual == expected, f"Q0.3 failed: a={a_float}, b={b_float}, expected={expected}, got={actual}"

            # Q0.7 (8-bit) output: after 2 cycles
            await RisingEdge(dut.clk)
            if hasattr(dut, "q0_7_valid") and dut.q0_7_valid.value:
                expected = expected_product_q0_signed(a, b, in_frac_bits, 7, 8)
                actual = int(dut.q0_7_out.value)
                assert actual == expected, f"Q0.7 failed: a={a_float}, b={b_float}, expected={expected}, got={actual}"

            # Q0.15 (16-bit) output: after 4 cycles
            await RisingEdge(dut.clk)
            await RisingEdge(dut.clk)
            if hasattr(dut, "q0_15_valid") and dut.q0_15_valid.value:
                expected = expected_product_q0_signed(a, b, in_frac_bits, 15, 16)
                actual = int(dut.q0_15_out.value)
                assert actual == expected, f"Q0.15 failed: a={a_float}, b={b_float}, expected={expected}, got={actual}"

            dut.valid_in.value = 0
            await RisingEdge(dut.clk)

    # Flush remaining pipeline
    for _ in range(5):
        await RisingEdge(dut.clk)

    # Final check to ensure valid signals are cleared
    if hasattr(dut, "q0_3_valid"):
        assert dut.q0_3_valid.value == 0, "Q0.3 valid should be 0 after flush"
    if hasattr(dut, "q0_7_valid"):
        assert dut.q0_7_valid.value == 0, "Q0.7 valid should be 0 after flush"
    if hasattr(dut, "q0_15_valid"):
        assert dut.q0_15_valid.value == 0, "Q0.15 valid should be 0 after flush"
