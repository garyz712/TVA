import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np
import random

# Parameters
M = 16
K = 16
N = 16
WIDTH = 16
MAX_VAL = (1 << (WIDTH - 1)) - 1
TOLERANCE = 0.001  # FP16 tolerance for Q1.15

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
# DUT reset
# ----------------------
async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    for i in range(M * K):
        dut.a_in[i].value = 0
    for i in range(K * N):
        dut.b_in[i].value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await Timer(20, units="ns")

# ----------------------
# Drive inputs
# ----------------------
async def drive_inputs(dut, a_flat, b_flat):
    for i in range(M * K):
        dut.a_in[i].value = int(a_flat[i])
    for i in range(K * N):
        dut.b_in[i].value = int(b_flat[i])
    dut.start.value = 1
    await RisingEdge(dut.clk)
    while dut.done.value != 1:
        await RisingEdge(dut.clk)
    dut.start.value = 0

# ----------------------
# Compute expected result in Q1.30
# ----------------------
def compute_expected(a, b):
    expected_out = np.zeros((M, N), dtype=float)
    for k in range(K):
        for i in range(M):
            raw_a = int(a[i, k])
            a_val = q1_15_to_real(raw_a)
            for j in range(N):
                raw_b = int(b[k, j])
                b_val = q1_15_to_real(raw_b)
                expected_out[i, j] += a_val * b_val
        expected_out = np.clip(expected_out, -2.0, 1.999999999)
    expected_q = np.vectorize(real_to_q1_30)(expected_out)
    return expected_q.astype(np.int32)

# ----------------------
# Generate test case
# ----------------------
def generate_test_case():
    a = [real_to_q1_15(random.uniform(-0.9, 0.9)) for _ in range(M * K)]
    b = [real_to_q1_15(random.uniform(-0.9, 0.9)) for _ in range(K * N)]
    a_np = np.array(a, dtype=np.int16).reshape(M, K)
    b_np = np.array(b, dtype=np.int16).reshape(K, N)
    return a_np, b_np

# ----------------------
# Run test and compare
# ----------------------
async def run_test(dut, a, b, test_name):
    dut._log.info(f"Starting test: {test_name}")
    
    await drive_inputs(dut, a.flatten(), b.flatten())
    
    while dut.done.value != 1:
        await RisingEdge(dut.clk)

    c_out = []
    for i in range(M * N):
        value = int(dut.c_out[i].value)
        if value & (1 << 31):
            value -= (1 << 32)
        c_out.append(value)

    c_expected = compute_expected(a, b).flatten()

    passed = True
    for i in range(M * N):
        actual = q1_30_to_real(c_out[i])
        expected = q1_30_to_real(c_expected[i])
        diff = actual - expected
        if abs(diff) > TOLERANCE:
            dut._log.error(f"Mismatch at c_out[{i}]: got {q1_30_to_real(c_out[i])}, expected {q1_30_to_real(c_expected[i])}")
            passed = False

    if passed:
        dut._log.info(f"Test {test_name} PASSED")
    else:
        dut._log.error(f"Test {test_name} FAILED")

    return passed

# ----------------------
# Main test function
# ----------------------
@cocotb.test()
async def test_matmul_array(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Test 1: Random matrices
    a, b = generate_test_case()
    await run_test(dut, a, b, "Random Matrices")

    # Test 2: Zero matrices
    a_zero = np.zeros((M, K), dtype=np.int16)
    b_zero = np.zeros((K, N), dtype=np.int16)
    await reset_dut(dut)
    await run_test(dut, a_zero, b_zero, "Zero Matrices")

    # Test 3: Identity-like A matrix (Q0.15 values)
    a_identity = np.array([
        real_to_q1_15(1.0) if i % (K + 1) == 0 else real_to_q1_15(0.0)
        for i in range(M * K)
    ], dtype=np.int16).reshape(M, K)
    b_random = np.array([
        real_to_q1_15(random.uniform(-0.9, 0.9))
        for _ in range(K * N)
    ], dtype=np.int16).reshape(K, N)
    await reset_dut(dut)
    await run_test(dut, a_identity, b_random, "Identity A Matrix")

    # Test 4: Small value matrices (to avoid overflow)
    a_small = np.array([
        real_to_q1_15(random.uniform(-0.01, 0.01))
        for _ in range(M * K)
    ], dtype=np.int16).reshape(M, K)
    b_small = np.array([
        real_to_q1_15(random.uniform(-0.01, 0.01))
        for _ in range(K * N)
    ], dtype=np.int16).reshape(K, N)
    await reset_dut(dut)
    await run_test(dut, a_small, b_small, "Small Values")

    dut._log.info("All tests completed.")
