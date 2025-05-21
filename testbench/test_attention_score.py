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
    while dut.done.value != 1:
        await RisingEdge(dut.clk)
    dut.start.value = 0

# ----------------------
# Compute expected result in Q1.15 (clipped to 16 bits)
# ----------------------
def compute_expected(q, k):
    # Reshape to (L*N, E) for Q and (L, N, E) for K
    q_np = q.reshape(L*N, E)
    k_np = k.reshape(L, N, E)
    # Transpose K to (E, N, L)
    k_transpose = np.transpose(k_np, (2, 1, 0))
    # Reshape K_transpose to (E, N*L) for matrix multiplication
    k_transpose = k_transpose.reshape(E, N*L)
    # Compute Q * K^T in floating point
    expected_out = np.zeros((L*N, L), dtype=float)
    for e in range(E):
        for i in range(L*N):
            for j in range(L*N):
                q_val = q1_15_to_real(q_np[i, e])
                k_val = q1_15_to_real(k_np[j, 0, e])  # N=1
                expected_out[i, j] += q_val * k_val
        expected_out = np.clip(expected_out, -2.0, 1.999999999)
    expected_out = np.clip(expected_out, -1.0, 0.999969)
    # Convert to Q1.15 (16-bit output)
    expected_q = np.vectorize(real_to_q1_30)(expected_out)
    return expected_q.astype(np.int32)

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
    
    while dut.done.value != 1:
        await RisingEdge(dut.clk)

    a_out = []
    for i in range(L * N * L):
        value = int(dut.A_out[i].value)
        if value & (1 << 15):
            value -= (1 << 16)
        a_out.append(value)

    a_expected = compute_expected(q, k).flatten()

    passed = True
    for i in range(L * N * L):
        actual = q1_15_to_real(a_out[i])
        expected = q1_30_to_real(a_expected[i])
        diff = actual - expected
        if abs(diff) > TOLERANCE:
            dut._log.error(f"Mismatch at A_out[{i}]: got {q1_15_to_real(a_out[i])}, expected {q1_15_to_real(a_expected[i])}")
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

    # Test 3: Identity-like Q matrix (Q0.15 values)
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

    # Test 4: Small value matrices (to avoid overflow)
    q_small = np.array([
        real_to_q1_15(random.uniform(-0.01, 0.01))
        for _ in range(L * N * E)
    ], dtype=np.int16).reshape(L, N, E)
    k_small = np.array([
        real_to_q1_15(random.uniform(-0.01, 0.01))
        for _ in range(L * N * E)
    ], dtype=np.int16).reshape(L, N, E)
    await reset_dut(dut)
    await run_test(dut, q_small, k_small, "Small Values")

    dut._log.info("All tests completed.")