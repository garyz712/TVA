import cocotb
from cocotb.triggers import FallingEdge, RisingEdge, Timer
from cocotb.clock import Clock
import random
import numpy as np

# Helper function to convert float to Q0.15
def float_to_q0_15(value):
    if value >= 1.0 or value < -1.0:
        raise ValueError("Value must be in range [-1.0, 1.0)")
    scaled = int(value * (2**15))
    if scaled > 32767:
        scaled = 32767
    elif scaled < -32768:
        scaled = -32768
    return scaled & 0xFFFF


# Helper function to convert Q1.30 to float
def q1_30_to_float(value):
    value = value & 0xFFFFFFFF
    if value & 0x80000000:  # If sign bit is set
        value = -((~value + 1) & 0xFFFFFFFF)  # Two's complement
    return float(value) / (2**30)

@cocotb.test()
async def test_mul1clk(dut):
    """Test the mul1clk module"""
    
    # Start clock (100 MHz)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset the module
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.a.value = 0
    dut.b.value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await Timer(20, units="ns")
    
    # Test cases: (a_float, b_float, expected_product)
    test_cases = [
        (0.5, 0.5, 0.25),           # Positive * Positive
        (-0.5, 0.5, -0.25),         # Negative * Positive
        (-0.5, -0.5, 0.25),         # Negative * Negative
        (0.0, 0.5, 0.0),            # Zero * Positive
        (0.999969482421875, 0.999969482421875, 0.99993896484375),  # Max positive
        (-0.999969482421875, -0.999969482421875, 0.99993896484375), # Min negative
        (0.1, 0.2, 0.02),           # Small values
        (0.0, 0.0, 0.0),            # Zero * Zero
    ]
    
    for a_float, b_float, expected_float in test_cases:
        # Convert inputs to Q0.15
        a_int = float_to_q0_15(a_float)
        b_int = float_to_q0_15(b_float)
        
        # Apply inputs
        await FallingEdge(dut.clk)
        dut.a.value = a_int
        dut.b.value = b_int

        print(a_int)
        print(b_int)
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0

        while dut.q1_30_valid.value != 1:
            await RisingEdge(dut.clk)

        
        result = dut.q1_30_out.value.signed_integer
        print(result)
        result_float = q1_30_to_float(result)
            
        # Allow small error due to fixed-point precision
        error = abs(result_float - expected_float)
        assert error < 1/(2**15), f"Failed: a={a_float}, b={b_float}, expected={expected_float}, got={result_float}"
    
    # Random tests
    for _ in range(50):
        a_q15 = random.randint(-2**15, 2**15-1)
        b_q15 = random.randint(-2**15, 2**15-1)
        expected_float = q1_30_to_float(a_q15 * b_q15)
        
        
        # Apply inputs
        dut.a.value = a_q15
        dut.b.value = b_q15
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0
        
        while dut.q1_30_valid.value != 1:
            await RisingEdge(dut.clk)
            dut.valid_in.value = 0
        
        # Check output
        result = dut.q1_30_out.value.signed_integer
        result_float = q1_30_to_float(result)
            
            
        # Allow small error due to fixed-point precision
        error = abs(result_float - expected_float)
        assert error < 1/(2**15), f"Failed random: a={a_float}, b={b_float}, expected={expected_float}, got={result_float}"
    
    # Test reset
    dut.rst_n.value = 0
    await Timer(20, units="ns")
    assert dut.q1_30_valid.value == 0, "Valid signal not cleared on reset"
    assert dut.q1_30_out.value == 0, "Output not cleared on reset"