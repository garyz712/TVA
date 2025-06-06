

# test_self_attention_top.py
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np
import random
import os

# Parameters matching self_attention_top
DATA_WIDTH = 16
L = 16           # Sequence length
E = 32           # Embedding dimension
N = 1           # Number of attention heads
OUT_DATA_WIDTH = 32  # Q2.30 for matmul outputs
TOLERANCE = 0.005    # Increased tolerance due to fixed-point approximations and sqrt(8) division

# ----------------------
# Fixed-point helpers (from submodules)
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

def real_to_q1_3(x):
    val = int(x * (2**3))
    if val > 7:
        val = 7
    if val < -8:
        val = -8
    return val & 0xF

def nibble_to_real_q1_3(n):
    n &= 0xF
    if n & 0x8:
        n -= 0x10
    return float(n) / (2**3)

def real_to_q1_7(x):
    val = int(x * (2**7))
    if val > 127:
        val = 127
    if val < -128:
        val = -128
    return val & 0xFF

# Constants for sqrt(8) division
SQRT_E = np.sqrt(32.0)  # ≈ 2.828
INV_SQRT_E = 1.0 / SQRT_E  # ≈ 0.354
INV_SQRT_E_Q15 = real_to_q1_15(INV_SQRT_E)  # Hardware constant (11,600 in decimal)

def hw_multiply_inv_sqrt8(matmul_result_q30):
    matmul_signed = np.array(matmul_result_q30).astype(np.int32)
    inv_sqrt8_signed = np.int16(INV_SQRT_E_Q15)
    mult_result = np.int64(matmul_signed) * np.int64(inv_sqrt8_signed)
    q30_result = (mult_result >> 15) & 0xFFFFFFFF
    if q30_result & 0x80000000:
        q30_result = q30_result - 0x100000000
    return np.int32(q30_result)

# ----------------------
# DUT reset
# ----------------------
async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    for i in range(L * E):
        dut.x_in[i].value = 0
    for i in range(E * E):
        dut.WQ_in[i].value = 0
        dut.WK_in[i].value = 0
        dut.WV_in[i].value = 0
        dut.WO_in[i].value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await Timer(20, units="ns")

# ----------------------
# Drive inputs
# ----------------------
async def drive_inputs(dut, x, wq, wk, wv, wo):
    for i in range(L * E):
        dut.x_in[i].value = int(x[i])
    for i in range(E * E):
        dut.WQ_in[i].value = int(wq[i])
        dut.WK_in[i].value = int(wk[i])
        dut.WV_in[i].value = int(wv[i])
        dut.WO_in[i].value = int(wo[i])
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for completion
    timeout = 10000
    while dut.done.value != 1 and timeout > 0:
        await RisingEdge(dut.clk)
        timeout -= 1
    assert timeout > 0, "Timeout waiting for done signal"
    assert dut.done.value == 1, "Done signal not asserted"
    assert dut.out_valid.value == 1, "Output valid signal not asserted"

# ----------------------
# Compute expected result (modified to return intermediates)
# ----------------------
def compute_expected(x, wq, wk, wv, wo):
    x_np = x.reshape(L * N, E)
    wq_np = wq.reshape(E, E)
    wk_np = wk.reshape(E, E)
    wv_np = wv.reshape(E, E)
    wo_np = wo.reshape(E, E)

    # Step 1: QKV generation
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

    q = matmul_expected(x_np, wq_np).reshape(L, N, E)
    k = matmul_expected(x_np, wk_np).reshape(L, N, E)
    v = matmul_expected(x_np, wv_np).reshape(L, N, E)

    # Step 2: Attention scores
    q_np = q.reshape(L * N, E)
    k_np = k.reshape(L, N, E)
    matmul_result = np.zeros((L * N, L), dtype=float)
    for e in range(E):
        for i in range(L * N):
            for j in range(L * N):
                q_val = q1_15_to_real(q_np[i, e])
                k_val = q1_15_to_real(k_np[j, 0, e])  # N=1
                matmul_result[i, j] += q_val * k_val
        matmul_result = np.clip(matmul_result, -2.0, 1.999999999)
    matmul_q30 = np.vectorize(real_to_q1_30)(matmul_result)
    a_q30 = np.array([hw_multiply_inv_sqrt8(matmul_q30[i, j]) for i in range(L * N) for j in range(L)], dtype=np.int32)

    a_q15 = []
    for q30_val in a_q30:
        sign_bit = (q30_val >> 31) & 1
        int_bit = (q30_val >> 30) & 1
        if sign_bit == int_bit:
            q15_val = (sign_bit << 15) | ((q30_val >> 15) & 0x7FFF)
        else:
            q15_val = 0x8000 if sign_bit else 0x7FFF
        if q15_val & 0x8000:
            q15_val = q15_val - 0x10000
        a_q15.append(q15_val)
    a = np.array(a_q15, dtype=np.int16).reshape(L, N, L)

    # Step 3: Softmax approximation
    a_float = np.vectorize(q1_15_to_real)(a)
    a_softmax = np.zeros((L, N, L), dtype=float)
    for n in range(N):
        for i in range(L):
            relu_row = np.maximum(a_float[i, n, :], 0)
            row_sum = np.sum(relu_row)
            if row_sum != 0:
                a_softmax[i, n, :] = relu_row / row_sum
            else:
                a_softmax[i, n, :] = 0
    a_softmax = np.clip(a_softmax, -1.0, 0.999969482421875)
    a_softmax_q15 = np.vectorize(real_to_q1_15)(a_softmax).reshape(L * N, L)

    # Step 4: Precision assignment
    a_sum = np.sum(a_softmax_q15, axis=0)
    token_precision = []
    for s in a_sum:
        if s < 16384:
            code = 0  # int4
        elif s < 32768:
            code = 1  # int8
        else:
            code = 2  # fp16
        token_precision.append(code)

    # Step 5: A*V multiplication
    av_out = np.zeros((L, E), dtype=float)
    for k_idx in range(L):
        prec = token_precision[k_idx]
        for i in range(L):
            raw_a = int(a_softmax_q15[i, k_idx])
            if prec == 0:
                a_val = nibble_to_real_q1_3(raw_a >> 12)
            elif prec == 1:
                a_val = float(np.array((raw_a & 0xFF00) >> 8).astype(np.int8)) / (2**7)
            else:
                a_val = q1_15_to_real(raw_a)
            for j in range(E):
                raw_v = int(v[k_idx, 0, j])  # N=1
                if prec == 0:
                    v_val = nibble_to_real_q1_3(raw_v >> 12)
                elif prec == 1:
                    v_val = float(np.array((raw_v & 0xFF00) >> 8).astype(np.int8)) / (2**7)
                else:
                    v_val = q1_15_to_real(raw_v)
                av_out[i, j] += a_val * v_val
                av_out[i, j] = np.clip(av_out[i, j], -2.0, 1.999999999)
    av_out = np.clip(av_out, -1.0, 0.999969482421875)
    av_q15 = np.vectorize(real_to_q1_15)(av_out)

    # Step 6: W_O multiplication
    expected_out = np.zeros((L, E), dtype=float)
    for k_idx in range(E):
        for i in range(L):
            raw_av = int(av_q15[i, k_idx])
            av_val = q1_15_to_real(raw_av)
            for j in range(E):
                raw_wo = int(wo_np[k_idx, j])
                wo_val = q1_15_to_real(raw_wo)
                expected_out[i, j] += av_val * wo_val
        expected_out = np.clip(expected_out, -2.0, 1.999999999)
    out_q30 = np.vectorize(real_to_q1_30)(expected_out)

    out_q15 = []
    for q30_val in out_q30.flatten():
        sign_bit = (q30_val >> 31) & 1
        int_bit = (q30_val >> 30) & 1
        if sign_bit == int_bit:
            q15_val = (sign_bit << 15) | ((q30_val >> 15) & 0x7FFF)
        else:
            q15_val = 0x8000 if sign_bit else 0x7FFF
        if q15_val & 0x8000:
            q15_val = q15_val - 0x10000
        out_q15.append(q15_val)
    out_q15 = np.array(out_q15, dtype=np.int16).reshape(L, E)

    return {
        'Q': q.reshape(L, E),
        'K': k.reshape(L, E),
        'V': v.reshape(L, E),
        'A': a.reshape(L, L),
        'A_softmax': a_softmax_q15.reshape(L, L),
        'token_precision': np.array(token_precision, dtype=np.int8),
        'AV': av_q15,
        'out': out_q15
    }

# ----------------------
# Generate test case
# ----------------------
def generate_test_case():
    x = np.array([real_to_q1_15(random.uniform(-1, 1-2**-15)) for _ in range(L * E)], dtype=np.int16).reshape(L * N, E)
    wq = np.array([real_to_q1_15(random.uniform(-1, 1-2**-15)) for _ in range(E * E)], dtype=np.int16).reshape(E, E)
    wk = np.array([real_to_q1_15(random.uniform(-1, 1-2**-15)) for _ in range(E * E)], dtype=np.int16).reshape(E, E)
    wv = np.array([real_to_q1_15(random.uniform(-1, 1-2**-15)) for _ in range(E * E)], dtype=np.int16).reshape(E, E)
    wo = np.array([real_to_q1_15(random.uniform(-1, 1-2**-15)) for _ in range(E * E)], dtype=np.int16).reshape(E, E)
    return x, wq, wk, wv, wo

# ----------------------
# Run test and compare with intermediate assertions
# ----------------------
async def run_test(dut, x, wq, wk, wv, wo, test_name, b=0):
    dut._log.info(f"Starting test: {test_name}")
    
    # Drive inputs and wait for completion
    await drive_inputs(dut, x.flatten(), wq.flatten(), wk.flatten(), wv.flatten(), wo.flatten())
    
    # Compute expected results (including intermediates)
    expected = compute_expected(x, wq, wk, wv, wo)
    
    # Helper function to read DUT array and handle sign extension
    def read_dut_array(dut_array, size, shape):
        result = []
        for i in range(size):
            value = int(dut_array[i].value)
            if value & (1 << 15):
                value -= (1 << 16)
            result.append(value)
        return np.array(result, dtype=np.int16).reshape(shape)

    # Check QKV generator outputs
    q_dut = read_dut_array(dut.qkv_inst.Q_out, L * E, (L, E))
    k_dut = read_dut_array(dut.qkv_inst.K_out, L * E, (L, E))
    v_dut = read_dut_array(dut.qkv_inst.V_out, L * E, (L, E))
    
    passed = True
    max_error = 0.0

    # QKV assertions
    for i in range(L):
        for j in range(E):
            q_actual = q1_15_to_real(q_dut[i, j])
            q_expected = q1_15_to_real(expected['Q'][i, j])
            diff = abs(q_actual - q_expected)
            max_error = max(max_error, diff)
            if diff > TOLERANCE:
                dut._log.error(f"QKV (Q) Mismatch at [{i},{j}]: got {q_actual:.6f}, expected {q_expected:.6f}, diff {diff:.6f}")
                passed = False

            k_actual = q1_15_to_real(k_dut[i, j])
            k_expected = q1_15_to_real(expected['K'][i, j])
            diff = abs(k_actual - k_expected)
            max_error = max(max_error, diff)
            if diff > TOLERANCE:
                dut._log.error(f"QKV (K) Mismatch at [{i},{j}]: got {k_actual:.6f}, expected {k_expected:.6f}, diff {diff:.6f}")
                passed = False

            v_actual = q1_15_to_real(v_dut[i, j])
            v_expected = q1_15_to_real(expected['V'][i, j])
            diff = abs(v_actual - v_expected)
            max_error = max(max_error, diff)
            if diff > TOLERANCE:
                dut._log.error(f"QKV (V) Mismatch at [{i},{j}]: got {v_actual:.6f}, expected {v_expected:.6f}, diff {diff:.6f}")
                passed = False

    # Check attention_score output
    a_dut = read_dut_array(dut.attn_score_inst.A_out, L * L, (L, L))
    for i in range(L):
        for j in range(L):
            a_actual = q1_15_to_real(a_dut[i, j])
            a_expected = q1_15_to_real(expected['A'][i, j])
            diff = abs(a_actual - a_expected)
            max_error = max(max_error, diff)
            if diff > TOLERANCE:
                dut._log.error(f"Attention Score Mismatch at [{i},{j}]: got {a_actual:.6f}, expected {a_expected:.6f}, diff {diff:.6f}")
                passed = False

    # Check softmax_approx output
    a_softmax_dut = read_dut_array(dut.softmax_inst.A_out, L * L, (L, L))
    for i in range(L):
        for j in range(L):
            a_softmax_actual = q1_15_to_real(a_softmax_dut[i, j])
            a_softmax_expected = q1_15_to_real(expected['A_softmax'][i, j])
            diff = abs(a_softmax_actual - a_softmax_expected)
            max_error = max(max_error, diff)
            if diff > TOLERANCE:
                dut._log.error(f"Softmax Mismatch at [{i},{j}]: got {a_softmax_actual:.6f}, expected {a_softmax_expected:.6f}, diff {diff:.6f}")
                passed = False

    # Check precision_assigner output
    token_precision_dut = []
    for i in range(L):
        value = int(dut.prec_assign_inst.token_precision[i].value)
        token_precision_dut.append(value)
    token_precision_dut = np.array(token_precision_dut, dtype=np.int8)
    for i in range(L):
        if token_precision_dut[i] != expected['token_precision'][i]:
            dut._log.error(f"Precision Assigner Mismatch at token {i}: got {token_precision_dut[i]}, expected {expected['token_precision'][i]}")
            passed = False

    # Check attention_av_multiply output
    av_dut = read_dut_array(dut.av_multiply_inst.out_mem, L * E, (L, E))
    for i in range(L):
        for j in range(E):
            av_actual = q1_15_to_real(av_dut[i, j])
            av_expected = q1_15_to_real(expected['AV'][i, j])
            diff = abs(av_actual - av_expected)
            max_error = max(max_error, diff)
            if diff > TOLERANCE:
                dut._log.error(f"AV Multiply Mismatch at [{i},{j}]: got {av_actual:.6f}, expected {av_expected:.6f}, diff {diff:.6f}")
                passed = False

    # Check final output
    out_dut = read_dut_array(dut.out, L * E, (L, E))
    os.makedirs(f"../verilog_outputs", exist_ok=True)
    np.save(f"../verilog_outputs/self_attention_out_{b}.npy", out_dut)
    for i in range(L):
        for j in range(E):
            out_actual = q1_15_to_real(out_dut[i, j])
            out_expected = q1_15_to_real(expected['out'][i, j])
            diff = abs(out_actual - out_expected)
            max_error = max(max_error, diff)
            if diff > TOLERANCE:
                dut._log.error(f"Final Output Mismatch at [{i},{j}]: got {out_actual:.6f}, expected {out_expected:.6f}, diff {diff:.6f}")
                passed = False
            else:
                dut._log.debug(f"Final Output [{i},{j}]: got {out_actual:.6f}, expected {out_expected:.6f}, diff {diff:.6f}")

    dut._log.info(f"Test {test_name}: Max error = {max_error:.6f}")
    if passed:
        dut._log.info(f"Test {test_name} PASSED")
    else:
        dut._log.error(f"Test {test_name} FAILED")
    
    return passed

@cocotb.test()
async def test_generate_for_colab(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    wq = np.load(f"../verilog_inputs/wq_np.npy").reshape(E, E).astype(np.int16)
    wk = np.load(f"../verilog_inputs/wk_np.npy").reshape(E, E).astype(np.int16)
    wv = np.load(f"../verilog_inputs/wv_np.npy").reshape(E, E).astype(np.int16)
    wo = np.load(f"../verilog_inputs/wo_np.npy").reshape(E, E).astype(np.int16)

    for b in range(10):
        x = np.load(f"../verilog_inputs/x_ln_np_{b}.npy").reshape(L*N, E).astype(np.int16)
        await run_test(dut, x, wq, wk, wv, wo, "Actual Input", b)
        await reset_dut(dut)


# ----------------------
# Main test function
# ----------------------
# @cocotb.test()
async def test_self_attention_top(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Test 1: Random matrices
    for _ in range(100):
        x, wq, wk, wv, wo = generate_test_case()
        await run_test(dut, x, wq, wk, wv, wo, "Random Matrices")
        await reset_dut(dut)

    # Test 2: Zero matrices
    x_zero = np.zeros((L * N, E), dtype=np.int16)
    wq_zero = np.zeros((E, E), dtype=np.int16)
    wk_zero = np.zeros((E, E), dtype=np.int16)
    wv_zero = np.zeros((E, E), dtype=np.int16)
    wo_zero = np.zeros((E, E), dtype=np.int16)
    await run_test(dut, x_zero, wq_zero, wk_zero, wv_zero, wo_zero, "Zero Matrices")
    await reset_dut(dut)

    # Test 3: Identity-like input matrix
    x_identity = np.array([
        real_to_q1_15(1.0) if i % (E + 1) == 0 else real_to_q1_15(0.0)
        for i in range(L * E)
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
    wo_random = np.array([
        real_to_q1_15(random.uniform(-0.9, 0.9))
        for _ in range(E * E)
    ], dtype=np.int16).reshape(E, E)
    await run_test(dut, x_identity, wq_random, wk_random, wv_random, wo_random, "Identity X Matrix")
    await reset_dut(dut)

    # Test 4: Small value matrices
    x_small = np.array([
        real_to_q1_15(random.uniform(-0.01, 0.01))
        for _ in range(L * E)
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
    wo_small = np.array([
        real_to_q1_15(random.uniform(-0.01, 0.01))
        for _ in range(E * E)
    ], dtype=np.int16).reshape(E, E)
    await run_test(dut, x_small, wq_small, wk_small, wv_small, wo_small, "Small Values")
    await reset_dut(dut)

    # Test 5: Maximum value matrices
    x_max = np.full((L * N, E), real_to_q1_15(0.9), dtype=np.int16)
    wq_max = np.full((E, E), real_to_q1_15(0.9), dtype=np.int16)
    wk_max = np.full((E, E), real_to_q1_15(0.9), dtype=np.int16)
    wv_max = np.full((E, E), real_to_q1_15(0.9), dtype=np.int16)
    wo_max = np.full((E, E), real_to_q1_15(0.9), dtype=np.int16)
    await run_test(dut, x_max, wq_max, wk_max, wv_max, wo_max, "Maximum Values")
    await reset_dut(dut)

    # Test 6: Mixed precision trigger
    x = np.array([real_to_q1_15(random.uniform(-0.5, 0.5)) for _ in range(L * E)], dtype=np.int16).reshape(L * N, E)
    wq = np.array([real_to_q1_15(random.uniform(-0.5, 0.5)) for _ in range(E * E)], dtype=np.int16).reshape(E, E)
    wk = np.array([real_to_q1_15(random.uniform(-0.5, 0.5)) for _ in range(E * E)], dtype=np.int16).reshape(E, E)
    wv = np.array([real_to_q1_15(random.uniform(-0.5, 0.5)) for _ in range(E * E)], dtype=np.int16).reshape(E, E)
    wo = np.array([real_to_q1_15(random.uniform(-0.5, 0.5)) for _ in range(E * E)], dtype=np.int16).reshape(E, E)
    await run_test(dut, x, wq, wk, wv, wo, "Mixed Precision Trigger")
    await reset_dut(dut)

    # Test 7: Orthogonal weight matrices
    wq_ortho = np.eye(E, dtype=float) * 0.9
    wk_ortho = np.eye(E, dtype=float) * 0.9
    wv_ortho = np.eye(E, dtype=float) * 0.9
    wo_ortho = np.eye(E, dtype=float) * 0.9
    x_random = np.array([real_to_q1_15(random.uniform(-0.9, 0.9)) for _ in range(L * E)], dtype=np.int16).reshape(L * N, E)
    wq_ortho = np.vectorize(real_to_q1_15)(wq_ortho).reshape(E, E)
    wk_ortho = np.vectorize(real_to_q1_15)(wk_ortho).reshape(E, E)
    wv_ortho = np.vectorize(real_to_q1_15)(wv_ortho).reshape(E, E)
    wo_ortho = np.vectorize(real_to_q1_15)(wo_ortho).reshape(E, E)
    await run_test(dut, x_random, wq_ortho, wk_ortho, wv_ortho, wo_ortho, "Orthogonal Weights")
    await reset_dut(dut)

    # Test 8: Sequential runs
    for i in range(3):
        x, wq, wk, wv, wo = generate_test_case()
        await run_test(dut, x, wq, wk, wv, wo, f"Sequential Run {i+1}")
        await reset_dut(dut)

    dut._log.info("All tests completed.")
















