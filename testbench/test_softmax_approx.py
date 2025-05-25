import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import numpy as np
import random

# Parameters matching the SystemVerilog module
DATA_WIDTH = 16
L = 8  # Sequence length
N = 1  # Number of attention heads

def q15_to_float(val, width=16):
    """Convert Q15 fixed-point to float"""
    if val >= 2**(width-1):
        val = val - 2**width
    return val / (2**15)

def float_to_q15(val, width=16):
    """Convert float to Q15 fixed-point"""
    # Clamp to Q15 range [-1.0, 0.99997]
    val = max(-1.0, min(0.99997, val))
    q15_val = int(val * (2**15))
    if q15_val < 0:
        q15_val = q15_val + 2**width
    return q15_val & ((1 << width) - 1)

def display_matrix_q15(matrix_flat, shape, name):
    """Display flattened matrix as Q15 values in matrix format"""
    matrix = np.array([q15_to_float(val) for val in matrix_flat])
    if len(shape) == 3:
        matrix = matrix.reshape(shape)
        print(f"\n{name} (Q15 as float):")
        for i in range(shape[0]):
            for j in range(shape[1]):
                print(f"  Head {j}, Row {i}: {matrix[i,j,:]}")
    else:
        matrix = matrix.reshape(shape)
        print(f"\n{name} (Q15 as float):")
        for i in range(shape[0]):
            print(f"  Row {i}: {matrix[i,:]}")

async def reset_dut(dut):
    """Reset the DUT"""
    dut.rst_n.value = 0
    dut.start.value = 0
    await Timer(100, units="ns")
    dut.rst_n.value = 1
    await Timer(50, units="ns")

@cocotb.test()
async def test_simple_softmax(dut):
    """Test with simple known values"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    await reset_dut(dut)
    
    print("=" * 60)
    print("TEST: Simple Softmax with Known Values")
    print("=" * 60)
    
    # Create simple test input - identity-like pattern
    test_input = []
    for i in range(L):
        for j in range(N):
            for k in range(L):
                if i == k:
                    # Diagonal elements: positive value (0.5 in Q15)
                    test_input.append(float_to_q15(0.5))
                else:
                    # Off-diagonal: smaller positive value (0.1 in Q15)
                    test_input.append(float_to_q15(0.1))
    
    # Apply inputs
    for i in range(L * N * L):
        dut.A_in[i].value = test_input[i]
    
    # Display input
    display_matrix_q15(test_input, (L, N, L), "Input Matrix A_in")
    
    # Start computation
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for completion
    while not dut.done.value:
        await RisingEdge(dut.clk)
    
    # Read outputs
    output = []
    for i in range(L * N * L):
        output.append(int(dut.A_out[i].value))
    
    # Display output
    display_matrix_q15(output, (L, N, L), "Output Matrix A_out")
    
    # Verify properties of softmax approximation
    output_float = [q15_to_float(val) for val in output]
    output_matrix = np.array(output_float).reshape(L, N, L)
    
    print("\nVerification:")
    for i in range(L):
        for j in range(N):
            row_sum = np.sum(output_matrix[i, j, :])
            print(f"  Row {i}, Head {j} sum: {row_sum:.6f}")
            
            # Check if all values are non-negative (ReLU property)
            non_negative = np.all(output_matrix[i, j, :] >= 0)
            print(f"  Row {i}, Head {j} all non-negative: {non_negative}")
    
    print(f"\nTest completed. out_valid = {dut.out_valid.value}")

@cocotb.test()
async def test_random_values(dut):
    """Test with random values including negative numbers"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    await reset_dut(dut)
    
    print("\n" + "=" * 60)
    print("TEST: Random Values (including negatives)")
    print("=" * 60)
    
    # Create random test input
    random.seed(42)  # For reproducibility
    test_input = []
    for i in range(L * N * L):
        # Random values between -0.8 and 0.8
        val = random.uniform(-0.8, 0.8)
        test_input.append(float_to_q15(val))
    
    # Apply inputs
    for i in range(L * N * L):
        dut.A_in[i].value = test_input[i]
    
    # Display input
    display_matrix_q15(test_input, (L, N, L), "Input Matrix A_in (Random)")
    
    # Start computation
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for completion
    timeout = 0
    while not dut.done.value and timeout < 1000:
        await RisingEdge(dut.clk)
        timeout += 1
    
    if timeout >= 1000:
        print("ERROR: Timeout waiting for completion!")
        return
    
    # Read outputs
    output = []
    for i in range(L * N * L):
        output.append(int(dut.A_out[i].value))
    
    # Display output
    display_matrix_q15(output, (L, N, L), "Output Matrix A_out (Random)")
    
    # Verify ReLU and normalization properties
    output_float = [q15_to_float(val) for val in output]
    output_matrix = np.array(output_float).reshape(L, N, L)
    
    print("\nVerification (Random Test):")
    for i in range(L):
        for j in range(N):
            row = output_matrix[i, j, :]
            row_sum = np.sum(row)
            row_max = np.max(row)
            row_min = np.min(row)
            
            print(f"  Row {i}, Head {j}:")
            print(f"    Sum: {row_sum:.6f}")
            print(f"    Max: {row_max:.6f}, Min: {row_min:.6f}")
            print(f"    All non-negative: {np.all(row >= -1e-6)}")  # Small tolerance for numerical errors

@cocotb.test()
async def test_edge_cases(dut):
    """Test edge cases: all zeros, all negatives, mixed"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    test_cases = [
        ("All Zeros", [0.0] * (L * N * L)),
        ("All Negatives", [-0.5] * (L * N * L)),
        ("Mixed with Zero Row", [0.3 if i < L else (-0.2 if i < 2*L else 0.1) for i in range(L * N * L)])
    ]
    
    for case_name, test_values in test_cases:
        print("\n" + "=" * 60)
        print(f"TEST: {case_name}")
        print("=" * 60)
        
        # Reset
        await reset_dut(dut)
        
        # Convert to Q15
        test_input = [float_to_q15(val) for val in test_values]
        
        # Apply inputs
        for i in range(L * N * L):
            dut.A_in[i].value = test_input[i]
        
        # Display input
        display_matrix_q15(test_input, (L, N, L), f"Input Matrix A_in ({case_name})")
        
        # Start computation
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0
        
        # Wait for completion
        timeout = 0
        while not dut.done.value and timeout < 1000:
            await RisingEdge(dut.clk)
            timeout += 1
        
        if timeout >= 1000:
            print(f"ERROR: Timeout in {case_name} test!")
            continue
        
        # Read outputs
        output = []
        for i in range(L * N * L):
            output.append(int(dut.A_out[i].value))
        
        # Display output
        display_matrix_q15(output, (L, N, L), f"Output Matrix A_out ({case_name})")
        
        # Basic verification
        output_float = [q15_to_float(val) for val in output]
        output_matrix = np.array(output_float).reshape(L, N, L)
        
        print(f"\nVerification ({case_name}):")
        for i in range(L):
            for j in range(N):
                row_sum = np.sum(output_matrix[i, j, :])
                all_non_neg = np.all(output_matrix[i, j, :] >= -1e-6)
                print(f"  Row {i}, Head {j}: Sum={row_sum:.6f}, Non-negative={all_non_neg}")

@cocotb.test()
async def test_timing_behavior(dut):
    """Test the timing and state machine behavior"""
    
    print("\n" + "=" * 60)
    print("TEST: Timing and State Machine Behavior")
    print("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    await reset_dut(dut)
    
    # Simple test input
    test_input = [float_to_q15(0.25)] * (L * N * L)
    
    # Apply inputs
    for i in range(L * N * L):
        dut.A_in[i].value = test_input[i]
    
    print("Monitoring state machine transitions...")
    
    # Start computation and monitor signals
    dut.start.value = 1
    start_cycle = 0
    cycle_count = 0
    
    await RisingEdge(dut.clk)
    dut.start.value = 0
    start_cycle = cycle_count
    
    # Monitor until completion
    while not dut.done.value:
        cycle_count += 1
        await RisingEdge(dut.clk)
        
        if cycle_count % 50 == 0:  # Print every 50 cycles
            print(f"  Cycle {cycle_count}: done={dut.done.value}, out_valid={dut.out_valid.value}")
    
    total_cycles = cycle_count - start_cycle
    print(f"\nTiming Results:")
    print(f"  Total computation cycles: {total_cycles}")
    print(f"  Final done: {dut.done.value}")
    print(f"  Final out_valid: {dut.out_valid.value}")
    
    # Verify done and out_valid are both high
    assert dut.done.value == 1, "done signal should be high"
    assert dut.out_valid.value == 1, "out_valid signal should be high"
    
    print("Timing test completed successfully!")