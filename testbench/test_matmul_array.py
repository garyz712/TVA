import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np
import random

# Parameters matching the Verilog module
M = 16
K = 16
N = 16
WIDTH = 16
MAX_VAL = (1 << (WIDTH - 1)) - 1  # Max signed value for WIDTH-bit numbers

async def reset_dut(dut):
    """Reset the DUT."""
    dut.rst_n.value = 0
    dut.start.value = 0
    for i in range(M * K):
        dut.a_in[i].value = 0
    for i in range(K * N):
        dut.b_in[i].value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await Timer(20, units="ns")

async def drive_inputs(dut, a_flat, b_flat):
    """Drive input matrices into the DUT."""
    for i in range(M * K):
        dut.a_in[i].value = a_flat[i]
    for i in range(K * N):
        dut.b_in[i].value = b_flat[i]
    dut.start.value = 1
    await RisingEdge(dut.clk)
    # Keep start high until done, then lower it
    while dut.done.value != 1:
        await RisingEdge(dut.clk)
    dut.start.value = 0

def compute_expected(a, b):
    """Compute expected matrix multiplication result using NumPy."""
    a_matrix = np.array(a, dtype=np.int64).reshape(M, K)
    b_matrix = np.array(b, dtype=np.int64).reshape(K, N)
    c_matrix = np.dot(a_matrix, b_matrix)
    return c_matrix.flatten()

def generate_test_case():
    """Generate random test matrices A and B."""
    a = [random.randint(-MAX_VAL, MAX_VAL) for _ in range(M * K)]
    b = [random.randint(-MAX_VAL, MAX_VAL) for _ in range(K * N)]
    return a, b

async def run_test(dut, a, b, test_name):
    """Run a single test case and verify output."""
    dut._log.info(f"Starting test: {test_name}")
    
    # Drive inputs
    await drive_inputs(dut, a, b)
    
    # Wait for done
    while dut.done.value != 1:
        await RisingEdge(dut.clk)
    
    # Read c_out
    c_out = [int(dut.c_out[i].value) for i in range(M * N)]
    
    # Compute expected result
    c_expected = compute_expected(a, b)
    
    # Verify
    passed = True
    for i in range(M * N):
        if c_out[i] != c_expected[i]:
            dut._log.error(f"Mismatch at c_out[{i}]: got {c_out[i]}, expected {c_expected[i]}")
            passed = False
    
    if passed:
        dut._log.info(f"Test {test_name} PASSED")
    else:
        dut._log.error(f"Test {test_name} FAILED")
    
    return passed

@cocotb.test()
async def test_matmul_array(dut):
    """Main test function for matmul_array module."""
    
    # Start clock (10ns period = 100MHz)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset DUT
    await reset_dut(dut)
    
    # Test 1: Random matrices
    a, b = generate_test_case()
    await run_test(dut, a, b, "Random Matrices")
    
    # Test 2: Zero matrices
    a_zero = [0] * (M * K)
    b_zero = [0] * (K * N)
    await reset_dut(dut)
    await run_test(dut, a_zero, b_zero, "Zero Matrices")
    
    # Test 3: Identity-like matrices (A = I, B = random)
    a_identity = [1 if i // K == i % K else 0 for i in range(M * K)]
    b_random = [random.randint(-MAX_VAL, MAX_VAL) for _ in range(K * N)]
    await reset_dut(dut)
    await run_test(dut, a_identity, b_random, "Identity A Matrix")
    
    # Test 4: Small values to avoid overflow
    a_small = [random.randint(-10, 10) for _ in range(M * K)]
    b_small = [random.randint(-10, 10) for _ in range(K * N)]
    await reset_dut(dut)
    await run_test(dut, a_small, b_small, "Small Values")
    
    dut._log.info("All tests completed")