
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np
import random

# Parameters
L = 8          # Sequence length
N = 1          # Batch size
E = 8          # Embedding dimension
WIDTH = 16     # Data width for inputs (Q1.15, 16-bit)
OUT_WIDTH = 16 # Output width (Q1.15, truncated from Q1.30)
MAX_VAL = (1 << (WIDTH - 1)) - 1
TOLERANCE = 0.001  # Tolerance for Q1.15 comparisons

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
        dut.x_in[i].value = 0
    for i in range(E * E):
        dut.WQ_in[i].value = 0
        dut.WK_in[i].value = 0
        dut.WV_in[i].value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await Timer(20, units="ns")

# ----------------------
# Drive inputs
# ----------------------
async def drive_inputs(dut, x, wq, wk, wv):
    for i in range(L * N * E):
        dut.x_in[i].value = int(x[i])
    for i in range(E * E):
        dut.WQ_in[i].value = int(wq[i])
        dut.WK_in[i].value = int(wk[i])
        dut.WV_in[i].value = int(wv[i])
    dut.start.value = 1
    await RisingEdge(dut.clk)
    while dut.done.value != 1:
        await RisingEdge(dut.clk)
    dut.start.value = 0

# ----------------------
# Compute expected result
# ----------------------
def compute_expected(x, wq, wk, wv):
    # Reshape inputs to appropriate dimensions
    x_np = x.reshape(L * N, E)
    wq_np = wq.reshape(E, E)
    wk_np = wk.reshape(E, E)
    wv_np = wv.reshape(E, E)

    # Helper function for matrix multiplication (from test_matmul_array.py)
    def matmul_expected(a, b):
        expected_out = np.zeros((L * N, E), dtype=float)
        for k in range(E):
            for i in range(L * N):
                raw_a = int(a[i, k])
                a_val = q1_15_to_real(raw_a)
                for j in range(E):
                    raw_b = int(b[k, j])
                    b_val = q1_15_to_real(raw_b)
                    expected_out[i, j] += a_val * b_val
            expected_out = np.clip(expected_out, -2.0, 1.999999999)
        expected_q30 = np.vectorize(real_to_q1_30)(expected_out)
        # Convert Q1.30 to Q1.15 with saturation (mimicking hardware truncation)
        expected_q15 = []
        for q30_val in expected_q30.flatten():
            sign_bit = (q30_val >> 31) & 1
            int_bit = (q30_val >> 30) & 1
            if sign_bit == int_bit:
                q15_val = (sign_bit << 15) | ((q30_val >> 15) & 0x7FFF)
            else:
                q15_val = 0x8000 if sign_bit else 0x7FFF
            if q15_val & 0x8000:
                q15_val = q15_val - 0x10000
            expected_q15.append(q15_val)
        return np.array(expected_q15, dtype=np.int16).reshape(L * N, E)

    # Compute Q = X * WQ, K = X * WK, V = X * WV
    q_expected = matmul_expected(x_np, wq_np)
    k_expected = matmul_expected(x_np, wk_np)
    v_expected = matmul_expected(x_np, wv_np)

    return q_expected, k_expected, v_expected

# ----------------------
# Generate test case
# ----------------------
def generate_test_case():
    x = [real_to_q1_15(random.uniform(-0.9, 0.9)) for _ in range(L * N * E)]
    wq = [real_to_q1_15(random.uniform(-0.9, 0.9)) for _ in range(E * E)]
    wk = [real_to_q1_15(random.uniform(-0.9, 0.9)) for _ in range(E * E)]
    wv = [real_to_q1_15(random.uniform(-0.9, 0.9)) for _ in range(E * E)]
    
    x_np = np.array(x, dtype=np.int16).reshape(L * N, E)
    wq_np = np.array(wq, dtype=np.int16).reshape(E, E)
    wk_np = np.array(wk, dtype=np.int16).reshape(E, E)
    wv_np = np.array(wv, dtype=np.int16).reshape(E, E)
    
    return x_np, wq_np, wk_np, wv_np

# ----------------------
# Run test and compare
# ----------------------
async def run_test(dut, x, wq, wk, wv, test_name):
    dut._log.info(f"Starting test: {test_name}")
    
    await drive_inputs(dut, x.flatten(), wq.flatten(), wk.flatten(), wv.flatten())
       
    while dut.done.value != 1:
        await RisingEdge(dut.clk)
        
    await RisingEdge(dut.clk)
    
    # Read outputs
    q_out = []
    k_out = []
    v_out = []
    for i in range(L * N * E):
        q_value = int(dut.Q_out[i].value)
        k_value = int(dut.K_out[i].value)
        v_value = int(dut.V_out[i].value)
        if q_value & (1 << 15):
            q_value -= (1 << 16)
        if k_value & (1 << 15):
            k_value -= (1 << 16)
        if v_value & (1 << 15):
            v_value -= (1 << 16)
        q_out.append(q_value)
        k_out.append(k_value)
        v_out.append(v_value)
    
    # Compute expected results
    q_expected, k_expected, v_expected = compute_expected(x, wq, wk, wv)
    q_expected = q_expected.flatten()
    k_expected = k_expected.flatten()
    v_expected = v_expected.flatten()
    
    # Compare results
    passed = True
    for i in range(L * N * E):
        q_actual = q1_15_to_real(q_out[i])
        q_exp = q1_15_to_real(q_expected[i])
        k_actual = q1_15_to_real(k_out[i])
        k_exp = q1_15_to_real(k_expected[i])
        v_actual = q1_15_to_real(v_out[i])
        v_exp = q1_15_to_real(v_expected[i])
        
        if abs(q_actual - q_exp) > TOLERANCE:
            dut._log.error(f"Mismatch at Q_out[{i}]: got {q_actual}, expected {q_exp}")
            passed = False
        if abs(k_actual - k_exp) > TOLERANCE:
            dut._log.error(f"Mismatch at K_out[{i}]: got {k_actual}, expected {k_exp}")
            passed = False
        if abs(v_actual - v_exp) > TOLERANCE:
            dut._log.error(f"Mismatch at V_out[{i}]: got {v_actual}, expected {v_exp}")
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
async def test_qkv_generator(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Test 1: Random matrices
    for _ in range(100):
        x, wq, wk, wv = generate_test_case()
        await reset_dut(dut)
        await run_test(dut, x, wq, wk, wv, "Random Matrices")

    # Test 2: Zero matrices
    x_zero = np.zeros((L * N, E), dtype=np.int16)
    wq_zero = np.zeros((E, E), dtype=np.int16)
    wk_zero = np.zeros((E, E), dtype=np.int16)
    wv_zero = np.zeros((E, E), dtype=np.int16)
    await reset_dut(dut)
    await run_test(dut, x_zero, wq_zero, wk_zero, wv_zero, "Zero Matrices")

    # Test 3: Identity-like X matrix
    x_identity = np.array([
        real_to_q1_15(1.0) if i % (E + 1) == 0 else real_to_q1_15(0.0)
        for i in range(L * N * E)
    ], dtype=np.int16).reshape(L * N, E)
    wq_random = np.array([
        real_to_q1_15(random.uniform(-0.9, 0.9))
        for _ in range(E * E)
    ], dtype=np.int16).reshape(E, E)
    wk_random = np.array([
        real_to_q1_15(random.uniform(-0.9, 0.9))
        for _ in range(E * E)
    ], dtype=np.int16).reshape(E, E)
    wv_random = np.array([
        real_to_q1_15(random.uniform(-0.9, 0.9))
        for _ in range(E * E)
    ], dtype=np.int16).reshape(E, E)
    await reset_dut(dut)
    await run_test(dut, x_identity, wq_random, wk_random, wv_random, "Identity X Matrix")

    # Test 4: Small value matrices (to avoid overflow)
    x_small = np.array([
        real_to_q1_15(random.uniform(-0.01, 0.01))
        for _ in range(L * N * E)
    ], dtype=np.int16).reshape(L * N, E)
    wq_small = np.array([
        real_to_q1_15(random.uniform(-0.01, 0.01))
        for _ in range(E * E)
    ], dtype=np.int16).reshape(E, E)
    wk_small = np.array([
        real_to_q1_15(random.uniform(-0.01, 0.01))
        for _ in range(E * E)
    ], dtype=np.int16).reshape(E, E)
    wv_small = np.array([
        real_to_q1_15(random.uniform(-0.01, 0.01))
        for _ in range(E * E)
    ], dtype=np.int16).reshape(E, E)
    await reset_dut(dut)
    await run_test(dut, x_small, wq_small, wk_small, wv_small, "Small Values")

    dut._log.info("All tests completed.")
