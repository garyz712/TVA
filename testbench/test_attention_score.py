import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np
import random

# Parameters
L = 8
N = 1
E = 8
WIDTH = 16
MAX_VAL = (1 << (WIDTH - 1)) - 1
TOLERANCE = 0.005  # Increased tolerance due to fixed-point approximation of 1/sqrt(8)

# Constants for sqrt(8) division
SQRT_8 = np.sqrt(8.0)  # ≈ 2.828
INV_SQRT_8 = 1.0 / SQRT_8  # ≈ 0.354
INV_SQRT_8_Q15 = 0x2D50  # Hardware constant (11,600 in decimal)

# ----------------------
# Fixed-point helpers
# ----------------------
def real_to_q1_15(x):
    val = int(x * (2**15))
    if val > 32767:
        val = 32767
    if val < -32768:
        val = -32768
    return np.array(val).astype(np.int16)

def q1_15_to_real(x):
    value = np.array(int(x) & 0xFFFF).astype(np.int16)
    return float(value) / (2**15)

def real_to_q1_30(x):
    val = int(x * (2**30))
    if val > (2**31 - 1):
        val = (2**31 - 1)
    if val < -(2**31):
        val = -(2**31)
    return np.array(val).astype(np.int32)

def q1_30_to_real(x):
    value = np.array(int(x) & 0xFFFFFFFF).astype(np.int32)
    return float(value) / (2**30)

# ----------------------
# Hardware-accurate fixed-point multiplication for 1/sqrt(8)
# ----------------------
def hw_multiply_inv_sqrt8(matmul_result_q30):
    """
    Simulate the hardware fixed-point multiplication:
    matmul_result (Q30) * INV_SQRT_8_Q15 (Q15) = Q45
    Then extract Q30 result by shifting right 15 bits
    """
    # Convert to signed 32-bit for multiplication
    matmul_signed = np.array(matmul_result_q30).astype(np.int32)
    inv_sqrt8_signed = np.int16(INV_SQRT_8_Q15)
    
    # 48-bit multiplication result
    mult_result = np.int64(matmul_signed) * np.int64(inv_sqrt8_signed)
    
    # Extract bits [46:15] to get Q30 result (equivalent to >> 15)
    q30_result = (mult_result >> 15) & 0xFFFFFFFF
    
    # Convert back to signed 32-bit
    if q30_result & 0x80000000:
        q30_result = q30_result - 0x100000000
    
    return np.int32(q30_result)

# ----------------------
# DUT reset
# ----------------------
async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    for i in range(L * N * E):
        dut.Q_in[i].value = 0
        dut.K_in[i].value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await Timer(20, units="ns")

# ----------------------
# Drive inputs
# ----------------------
async def drive_inputs(dut, q_flat, k_flat):
    for i in range(L * N * E):
        dut.Q_in[i].value = int(q_flat[i])
        dut.K_in[i].value = int(k_flat[i])
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for completion
    while dut.done.value != 1:
        await RisingEdge(dut.clk)

# ----------------------
# Compute expected result with sqrt(8) division
# ----------------------
def compute_expected(q, k):
    # Reshape to (L*N, E) for Q and (L, N, E) for K
    q_np = q.reshape(L*N, E)
    k_np = k.reshape(L, N, E)
    
    # Transpose K to (E, N, L)
    k_transpose = np.transpose(k_np, (2, 1, 0))
    k_transpose = k_transpose.reshape(E, N*L)
    
    # Compute Q * K^T in floating point (before scaling)
    matmul_result = np.zeros((L*N, L), dtype=float)
    for e in range(E):
        for i in range(L*N):
            for j in range(L*N):
                q_val = q1_15_to_real(q_np[i, e])
                k_val = q1_15_to_real(k_np[j, 0, e])  # N=1
                matmul_result[i, j] += q_val * k_val
    
    # Clip matmul result to 32-bit range
    matmul_result = np.clip(matmul_result, -2.0, 1.999999999)
    
    # Convert to Q1.30 for hardware-accurate division simulation
    matmul_q30 = np.vectorize(real_to_q1_30)(matmul_result)
    
    # Apply hardware-accurate division by sqrt(8)
    divided_results = []
    for i in range(L*N):
        for j in range(L):
            hw_result = hw_multiply_inv_sqrt8(matmul_q30[i, j])
            divided_results.append(hw_result)
    
    return np.array(divided_results, dtype=np.int32)

# ----------------------
# Generate test case
# ----------------------
def generate_test_case():
    q = [real_to_q1_15(random.uniform(-0.9, 0.9)) for _ in range(L * N * E)]
    k = [real_to_q1_15(random.uniform(-0.9, 0.9)) for _ in range(L * N * E)]
    q_np = np.array(q, dtype=np.int16).reshape(L, N, E)
    k_np = np.array(k, dtype=np.int16).reshape(L, N, E)
    return q_np, k_np

# ----------------------
# Run test and compare
# ----------------------
async def run_test(dut, q, k, test_name):
    dut._log.info(f"Starting test: {test_name}")
    
    await drive_inputs(dut, q.flatten(), k.flatten())
    
    # Read outputs
    a_out = []
    for i in range(L * N * L):
        value = int(dut.A_out[i].value)
        if value & (1 << 15):
            value -= (1 << 16)
        a_out.append(value)

    # Compute expected results
    a_expected_q30 = compute_expected(q, k)
    
    # Convert expected Q30 results to Q15 for comparison (simulate hardware truncation)
    a_expected_q15 = []
    for q30_val in a_expected_q30:
        # Simulate hardware truncation logic from divided_out to A_out
        sign_bit = (q30_val >> 31) & 1
        int_bit = (q30_val >> 30) & 1
        
        if sign_bit == int_bit:
            # In range: take sign bit and top 15 fractional bits
            q15_val = (sign_bit << 15) | ((q30_val >> 15) & 0x7FFF)
        else:
            # Out of range: saturate
            q15_val = 0x8000 if sign_bit else 0x7FFF
        
        # Convert to signed 16-bit
        if q15_val & 0x8000:
            q15_val = q15_val - 0x10000
        a_expected_q15.append(q15_val)

    # Compare results
    passed = True
    max_error = 0.0
    for i in range(L * N * L):
        actual = q1_15_to_real(a_out[i])
        expected = q1_15_to_real(a_expected_q15[i])
        diff = abs(actual - expected)
        max_error = max(max_error, diff)
        
        if diff > TOLERANCE:
            dut._log.error(f"Mismatch at A_out[{i}]: got {actual:.6f}, expected {expected:.6f}, diff {diff:.6f}")
            passed = False
        else:
            dut._log.debug(f"A_out[{i}]: got {actual:.6f}, expected {expected:.6f}, diff {diff:.6f}")

    dut._log.info(f"Test {test_name}: Max error = {max_error:.6f}")
    
    if passed:
        dut._log.info(f"Test {test_name} PASSED")
    else:
        dut._log.error(f"Test {test_name} FAILED")

    return passed

# ----------------------
# Main test function
# ----------------------
@cocotb.test()
async def test_attention_score(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Test 1: Random matrices
    q, k = generate_test_case()
    await run_test(dut, q, k, "Random Matrices")

    # Test 2: Zero matrices
    q_zero = np.zeros((L, N, E), dtype=np.int16)
    k_zero = np.zeros((L, N, E), dtype=np.int16)
    await reset_dut(dut)
    await run_test(dut, q_zero, k_zero, "Zero Matrices")

    # Test 3: Identity-like Q matrix
    q_identity = np.array([
        real_to_q1_15(1.0) if i % (E + 1) == 0 else real_to_q1_15(0.0)
        for i in range(L * N * E)
    ], dtype=np.int16).reshape(L, N, E)
    k_random = np.array([
        real_to_q1_15(random.uniform(-0.9, 0.9))
        for _ in range(L * N * E)
    ], dtype=np.int16).reshape(L, N, E)
    await reset_dut(dut)
    await run_test(dut, q_identity, k_random, "Identity Q Matrix")

    # Test 4: Small value matrices (to test precision)
    q_small = np.array([
        real_to_q1_15(random.uniform(-0.1, 0.1))
        for _ in range(L * N * E)
    ], dtype=np.int16).reshape(L, N, E)
    k_small = np.array([
        real_to_q1_15(random.uniform(-0.1, 0.1))
        for _ in range(L * N * E)
    ], dtype=np.int16).reshape(L, N, E)
    await reset_dut(dut)
    await run_test(dut, q_small, k_small, "Small Values")

    # Test 5: Maximum positive values (stress test)
    q_max = np.full((L, N, E), real_to_q1_15(0.9), dtype=np.int16)
    k_max = np.full((L, N, E), real_to_q1_15(0.9), dtype=np.int16)
    await reset_dut(dut)
    await run_test(dut, q_max, k_max, "Maximum Values")

    dut._log.info("All tests completed.")