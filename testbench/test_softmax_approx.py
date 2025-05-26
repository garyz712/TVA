import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

# Helper function to convert float to Q15
def float_to_q15(value):
    # Q15: 1 sign bit, 15 fractional bits
    # Clamp to [-1, 1) range for Q15
    if value >= 1.0:
        value = 1.0 - 1/32768
    elif value < -1.0:
        value = -1.0
    # Convert to Q15
    q15_value = int(round(value * 2**15))
    # Ensure 16-bit signed representation
    return q15_value & 0xFFFF

# Helper function to convert Q15 to float
def q15_to_float(value):
    # Handle sign extension for 16-bit value
    if value & 0x8000:
        value = value - 0x10000
    return value / 2**15

# Reference softmax approximation in Python
def reference_softmax_approx(input_data, L, N):
    # Reshape input to (L, N, L)
    data = np.array(input_data).reshape(L, N, L)
    output = np.zeros((L, N, L), dtype=float)
    
    for n in range(N):
        for i in range(L):
            # Apply ReLU
            relu_row = np.maximum(data[i, n, :], 0)
            # Compute sum of row
            row_sum = np.sum(relu_row)
            # Normalize
            if row_sum != 0:
                # Approximate normalization as in HDL: scale by 2^15 and divide
                output[i, n, :] = (relu_row) / row_sum
            else:
                output[i, n, :] = 0
    
    # Convert output to Q15
    output_q15 = [float_to_q15(x) for x in output.flatten()]
    return output_q15

@cocotb.test()
async def test_softmax_approx(dut):
    # Parameters
    DATA_WIDTH = 16
    L = 8
    N = 1
    ARRAY_SIZE = L * N * L
    
    # Start clock (10ns period)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset the DUT
    dut.rst_n.value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    # Test case 1: Random inputs in [-0.5, 0.5)
    np.random.seed(42)
    input_floats = np.random.uniform(-0.5, 0.5, ARRAY_SIZE)
    input_q15 = [float_to_q15(x) for x in input_floats]
    
    # Drive inputs
    dut.start.value = 0
    for i in range(ARRAY_SIZE):
        dut.A_in[i].value = input_q15[i]
    await RisingEdge(dut.clk)
    
    # Start the computation
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for done signal
    timeout_cycles = 1000
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            break
    else:
        raise Exception("Timeout waiting for done signal")
    
    # Check output valid
    assert dut.out_valid.value == 1, "Output valid signal not asserted"
    
    # Read outputs
    output_q15 = [int(dut.A_out[i].value) for i in range(ARRAY_SIZE)]
    output_floats = [q15_to_float(x) for x in output_q15]
    
    # Compute reference output
    ref_output_q15 = reference_softmax_approx(input_floats, L, N)
    ref_output_floats = [q15_to_float(x) for x in ref_output_q15]
    
    # Compare outputs with tolerance
    tolerance = 1/2**14  # Allow for 1/4 LSB error in Q15
    for i in range(ARRAY_SIZE):
        error = abs(output_floats[i] - ref_output_floats[i])
        assert error < tolerance, f"Output mismatch at index {i}: got {output_floats[i]}, expected {ref_output_floats[i]}"
    
    # Test case 2: All zeros
    input_q15 = [0] * ARRAY_SIZE
    dut.rst_n.value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    for i in range(ARRAY_SIZE):
        dut.A_in[i].value = input_q15[i]
    await RisingEdge(dut.clk)
    
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            break
    else:
        raise Exception("Timeout waiting for done signal")
    
    assert dut.out_valid.value == 1, "Output valid signal not asserted"
    
    output_q15 = [int(dut.A_out[i].value) for i in range(ARRAY_SIZE)]
    for i in range(ARRAY_SIZE):
        assert output_q15[i] == 0, f"Expected zero output for zero input at index {i}, got {output_q15[i]}"
    
    # Test case 3: Mixed positive and negative inputs
    input_floats = np.array([0.2, -0.3, 0.4, -0.1, 0.25, -0.2, 0.15, -0.4] * (L * N))
    input_q15 = [float_to_q15(x) for x in input_floats]
    
    dut.rst_n.value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    for i in range(ARRAY_SIZE):
        dut.A_in[i].value = input_q15[i]
    await RisingEdge(dut.clk)
    
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            break
    else:
        raise Exception("Timeout waiting for done signal")
    
    assert dut.out_valid.value == 1, "Output valid signal not asserted"
    
    output_q15 = [int(dut.A_out[i].value) for i in range(ARRAY_SIZE)]
    output_floats = [q15_to_float(x) for x in output_q15]
    
    ref_output_q15 = reference_softmax_approx(input_floats, L, N)
    ref_output_floats = [q15_to_float(x) for x in ref_output_q15]
    
    for i in range(ARRAY_SIZE):
        error = abs(output_floats[i] - ref_output_floats[i])
        assert error < tolerance, f"Output mismatch at index {i}: got {output_floats[i]}, expected {ref_output_floats[i]}"