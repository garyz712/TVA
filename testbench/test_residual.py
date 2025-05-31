import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np
import random

# Parameters matching the Verilog module
DATA_WIDTH = 16
SEQ_LEN = 8
EMB_DIM = 8
TOLERANCE = 0.0001  # Tolerance for Q1.15 comparisons

# Helper functions
def real_to_q1_15(x):
    """Convert real number to Q1.15 format (16-bit signed integer)."""
    val = int(x * (2**15))
    if val > 32767:  # 0x7FFF
        val = 32767
    if val < -32768:  # 0x8000
        val = -32768
    return np.array(val).astype(np.int16)

def q1_15_to_real(x):
    """Convert Q1.15 format to real number."""
    value = np.array(int(x) & 0xFFFF).astype(np.int16)
    return float(value) / (2**15)

def sat_add16(a, b):
    """Python model of Q1.15 saturating adder."""
    # Use int32 to avoid overflow warnings
    sum_val = np.int32(a) + np.int32(b)
    ovf = (a >= 0 and b >= 0 and sum_val < 0) or (a < 0 and b < 0 and sum_val >= 0)
    if not ovf:
        # Clamp to Q1.15 range
        if sum_val > 32767:
            return np.int16(32767)
        if sum_val < -32768:
            return np.int16(-32768)
        return np.int16(sum_val)
    elif a >= 0:  # Positive overflow
        return np.int16(32767)  # 0x7FFF
    else:  # Negative overflow
        return np.int16(-32768)  # 0x8000

def flatten_vector(vec, seq_len, emb_dim):
    """Flatten 2D vector to 1D for DUT input/output."""
    flat = np.zeros(seq_len * emb_dim, dtype=np.int16)
    for r in range(seq_len):
        for c in range(emb_dim):
            flat[r * emb_dim + c] = vec[r, c]
    return flat

def unflatten_vector(flat, seq_len, emb_dim):
    """Unflatten 1D vector to 2D for verification."""
    vec = np.zeros((seq_len, emb_dim), dtype=np.int16)
    for r in range(seq_len):
        for c in range(emb_dim):
            vec[r, c] = flat[r * emb_dim + c]
    return vec

def array_to_int(arr, bit_width):
    """Convert array of integers to a single integer (LSB-first)."""
    result = 0
    for i, val in enumerate(arr):  # LSB-first: [0,0] at lowest bits
        result |= (int(val) & ((1 << bit_width) - 1)) << (i * bit_width)
    return result

def extract_bits(signal, start, width):
    """Extract 'width' bits from 'signal' starting at 'start'."""
    mask = (1 << width) - 1
    shifted = signal >> start
    value = shifted & mask
    # Sign-extend if necessary
    if value & (1 << (width - 1)):
        value = value - (1 << width)
    return value

async def reset_dut(dut):
    """Reset the DUT."""
    dut.rst_n.value = 0
    dut.start.value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await Timer(10, units="ns")

async def run_test_case(dut, x_in, sub_in, test_name):
    """Run a single test case."""
    dut._log.info(f"\n{test_name}")
    
    # Flatten inputs
    x_in_flat = flatten_vector(x_in, SEQ_LEN, EMB_DIM)
    sub_in_flat = flatten_vector(sub_in, SEQ_LEN, EMB_DIM)
    
    # Log input values for debugging
    dut._log.info(f"x_in_flat (Q1.15 real): {[q1_15_to_real(x) for x in x_in_flat]}")
    dut._log.info(f"sub_in_flat (Q1.15 real): {[q1_15_to_real(x) for x in sub_in_flat]}")
    
    # Drive inputs
    x_in_int = array_to_int(x_in_flat, DATA_WIDTH)
    sub_in_int = array_to_int(sub_in_flat, DATA_WIDTH)
    dut.x_in.value = x_in_int
    dut.sub_in.value = sub_in_int
    dut._log.info(f"x_in integer: {x_in_int:#x}")
    dut._log.info(f"sub_in integer: {sub_in_int:#x}")
    dut._log.info("Inputs assigned successfully!")

    # Compute expected output
    expected_out = compute_expected(x_in, sub_in)
    expected_out_flat = flatten_vector(expected_out, SEQ_LEN, EMB_DIM)
    dut._log.info(f"Expected y_out_flat (Q1.15 real): {[q1_15_to_real(x) for x in expected_out_flat]}")
    
    # Start the DUT
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await RisingEdge(dut.done)
    await Timer(10, units="ns")
    
    # Verify output
    errors = 0
    y_out_value = int(dut.y_out.value)  # Get the full output as an integer
    dut._log.info(f"y_out integer: {y_out_value:#x}")
    y_out_flat = np.zeros(SEQ_LEN * EMB_DIM, dtype=np.int16)
    for i in range(SEQ_LEN * EMB_DIM):
        y_out_flat[i] = extract_bits(y_out_value, i * DATA_WIDTH, DATA_WIDTH)
    y_out = unflatten_vector(y_out_flat, SEQ_LEN, EMB_DIM)
    
    for r in range(SEQ_LEN):
        for c in range(EMB_DIM):
            actual = q1_15_to_real(y_out[r, c])
            expected = q1_15_to_real(expected_out[r, c])
            if abs(actual - expected) > TOLERANCE:
                dut._log.error(f"Error at [{r},{c}]: actual={actual}, expected={expected}, actual_raw={y_out[r,c]:#x}, expected_raw={expected_out[r,c]:#x}")
                errors += 1
    
    if errors == 0:
        dut._log.info(f"{test_name} passed!")
    else:
        dut._log.error(f"{test_name} failed with {errors} errors.")
    return errors

def compute_expected(x_in, sub_in):
    """Compute expected output using saturating addition."""
    expected = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.int16)
    for r in range(SEQ_LEN):
        for c in range(EMB_DIM):
            expected[r, c] = sat_add16(x_in[r, c], sub_in[r, c])
    return expected

@cocotb.test()
async def test_residual(dut):
    """Main test function for the residual module."""
    # Start clock (100 MHz)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Initialize DUT
    await reset_dut(dut)
    
    # Test Case 1: Fixed Inputs (for debugging)
    x_in = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.int16)
    sub_in = np.zeros((SEQ_LEN, EMB_DIM), dtype=np.int16)
    for r in range(SEQ_LEN):
        for c in range(EMB_DIM):
            x_in[r, c] = real_to_q1_15(0.5 if (r + c) % 2 == 0 else -0.5)
            sub_in[r, c] = real_to_q1_15(0.2 if (r + c) % 2 == 0 else -0.2)
    errors = await run_test_case(dut, x_in, sub_in, "Test Case 1: Fixed Inputs")
    assert errors == 0, f"Test Case 1 failed with {errors} errors"
    
    # Reset DUT
    await reset_dut(dut)
    
    # Test Case 2: Positive Saturation
    for r in range(SEQ_LEN):
        for c in range(EMB_DIM):
            x_in[r, c] = real_to_q1_15(0.8)
            sub_in[r, c] = real_to_q1_15(0.8)  # Sum > 1.0, should saturate to 0.999969
    errors = await run_test_case(dut, x_in, sub_in, "Test Case 2: Positive Saturation")
    assert errors == 0, f"Test Case 2 failed with {errors} errors"
    
    # Reset DUT
    await reset_dut(dut)
    
    # Test Case 3: Negative Saturation
    for r in range(SEQ_LEN):
        for c in range(EMB_DIM):
            x_in[r, c] = real_to_q1_15(-0.8)
            sub_in[r, c] = real_to_q1_15(-0.8)  # Sum < -1.0, should saturate to -1.0
    errors = await run_test_case(dut, x_in, sub_in, "Test Case 3: Negative Saturation")
    assert errors == 0, f"Test Case 3 failed with {errors} errors"
    
    # Reset DUT
    await reset_dut(dut)
    
    # Test Case 4: Zero Inputs
    for r in range(SEQ_LEN):
        for c in range(EMB_DIM):
            x_in[r, c] = real_to_q1_15(0.0)
            sub_in[r, c] = real_to_q1_15(0.0)
    errors = await run_test_case(dut, x_in, sub_in, "Test Case 4: Zero Inputs")
    assert errors == 0, f"Test Case 4 failed with {errors} errors"
    
    # Reset DUT
    await reset_dut(dut)
    
    # Test Case 5: Mixed Inputs
    for r in range(SEQ_LEN):
        for c in range(EMB_DIM):
            x_in[r, c] = real_to_q1_15(0.5 if (r + c) % 2 == 0 else -0.5)
            sub_in[r, c] = real_to_q1_15(-0.5 if (r + c) % 2 == 0 else 0.5)
    errors = await run_test_case(dut, x_in, sub_in, "Test Case 5: Mixed Inputs")
    assert errors == 0, f"Test Case 5 failed with {errors} errors"
    
    dut._log.info("All tests completed successfully.")
